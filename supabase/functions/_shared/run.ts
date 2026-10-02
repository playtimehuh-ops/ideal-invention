// Shared server-side runtime: model loading, credit accounting, execution. Nothing here ever reaches the browser.
import { createClient } from "npm:@supabase/supabase-js@2";
import { NovaModel, BPETokenizer } from "./engine.js";

export const admin = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!, { auth: { persistSession: false } });
export class HttpErr extends Error { constructor(public status: number, public code: string, msg: string) { super(msg); } }

export async function sha256(s: string) {
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s)))].map((b) => b.toString(16).padStart(2, "0")).join("");
}
function fnv1a(str: string) { let h = 0x811c9dc5; for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = (h * 0x01000193) >>> 0; } return h.toString(16); }

let cfgAt = 0, cfgCache: Record<string, string> = {};
export async function config() {
  if (Date.now() - cfgAt > 60_000) {
    const { data } = await admin.from("deploy_app_config").select("key,value");
    cfgCache = Object.fromEntries((data ?? []).map((r: any) => [r.key, r.value])); cfgAt = Date.now();
  }
  return cfgCache;
}

// Validate an imported model and load it into the real engine.
export function buildModel(o: any) {
  const bad = (m: string) => new HttpErr(400, "invalid_model", m);
  if (!o || !o.weights || !o.tokenizer || !o.config || !o.vocabSize) throw bad("Missing weights, tokenizer, config or vocabSize");
  if (o.checksum && o.checksum !== fnv1a(JSON.stringify(o.weights) + JSON.stringify(o.tokenizer))) throw bad("Checksum mismatch: file corrupted or altered");
  const t: any = BPETokenizer.fromJSON(o.tokenizer), m: any = new NovaModel(o.vocabSize, o.config), d = m.cfg.dModel;
  if (t.size !== o.vocabSize || o.weights.wte?.length !== o.vocabSize * d || o.weights.wpe?.length !== m.cfg.ctxLen * d ||
      o.weights.layers?.length !== m.cfg.nLayers) throw bad("Weight shapes do not match the model config");
  m.loadWeights(o.weights); m.eosId = t.vocab["<EOS>"];
  return { m, t };
}

const cache = new Map<string, { sha: string; m: any; t: any }>();
export async function loadModel(row: { id: string; owner: string; sha: string }) {
  const hit = cache.get(row.id);
  if (hit && hit.sha === row.sha) return hit;
  const { data, error } = await admin.storage.from("deploy_models").download(`${row.owner}/${row.id}.json`);
  if (error || !data) throw new HttpErr(500, "weights_unavailable", "Model weights could not be loaded");
  const text = await data.text();
  if (await sha256(text) !== row.sha) throw new HttpErr(500, "integrity", "Stored weights failed the integrity check");
  const e = { sha: row.sha, ...buildModel(JSON.parse(text)) };
  cache.set(row.id, e); if (cache.size > 3) cache.delete(cache.keys().next().value!);
  return e;
}

type Msg = { role: string; content: string };
// Order: authenticated (caller) → subscription → permission → cost → atomic reserve → execute → settle → usage record.
export async function runChat(a: { userId: string; modelId: string; messages: Msg[]; maxTokens?: number; temperature?: number; via: "playground" | "api"; keyId?: string }) {
  const { data: ok } = await admin.rpc("deploy_has_access", { p: a.userId });
  if (!ok) throw new HttpErr(402, "subscription_required", "An active subscription is required.");
  const { data: row } = await admin.from("deploy_models").select("id,owner,sha,settings,deployed").eq("id", a.modelId).maybeSingle();
  if (!row || row.owner !== a.userId) throw new HttpErr(404, "model_not_found", "Model not found.");   // same answer for "not yours" and "doesn't exist"
  if (a.via === "api" && !row.deployed) throw new HttpErr(403, "not_deployed", "This model is not deployed.");

  const msgs = a.messages;
  if (!Array.isArray(msgs) || !msgs.length || msgs.length > 50 || msgs[msgs.length - 1]?.role !== "user" ||
      msgs.some((x) => typeof x?.content !== "string" || (x.role !== "user" && x.role !== "assistant")))
    throw new HttpErr(400, "bad_messages", "messages must be a list ending with a user message.");
  const last = msgs[msgs.length - 1].content.trim().slice(0, 2000);
  if (!last) throw new HttpErr(400, "bad_messages", "Empty message.");

  const { m, t } = await loadModel(row);
  const st = row.settings ?? {}, cfg = await config();
  const maxTok = Math.max(1, Math.min(Number(a.maxTokens) || st.maxTok || 80, 300));
  const temp = Math.max(0.1, Math.min(Number(a.temperature) || st.temp || 0.8, 2));
  // Same prompt format the model was trained and tested with on the training site.
  const prompt = msgs.slice(0, -1).slice(-6).map((x) => (x.role === "user" ? "User: " : "Nova: ") + x.content.slice(0, 2000)).join("\n") + "\nUser: " + last + "\nNova:";
  const ids = t.encode(prompt);
  const cin = Number(cfg.credits_per_input_token ?? 1), cout = Number(cfg.credits_per_output_token ?? 1);
  const reserve = ids.length * cin + maxTok * cout;                       // worst-case cost, known before running
  const { data: alloc } = await admin.rpc("deploy_reserve_credits", { p_user: a.userId, p_amount: reserve });
  if (alloc == null) throw new HttpErr(402, "insufficient_credits", "Not enough credits for this request.");

  let out: number[], ms: number;
  try {
    const t0 = performance.now();
    out = m.generate(ids, maxTok, temp, st.topK || 30, st.topP || 0.9, { stop: false });
    ms = Math.round(performance.now() - t0);
  } catch (_e) {
    await admin.rpc("deploy_release_credits", { p_alloc: alloc, p_amount: reserve });   // failed runs cost nothing
    throw new HttpErr(500, "execution_failed", "Model execution failed.");
  }
  let reply = t.decode(out.slice(ids.length)); const cut = reply.indexOf("\nUser:"); if (cut >= 0) reply = reply.slice(0, cut);
  const outTok = out.length - ids.length, actual = ids.length * cin + outTok * cout;
  const { data: remaining } = await admin.rpc("deploy_settle_credits", { p_alloc: alloc, p_user: a.userId, p_reserved: reserve, p_actual: actual,
    p_model: row.id, p_via: a.via, p_key: a.keyId ?? null, p_in: ids.length, p_out: outTok, p_ms: ms });
  return { reply: reply.trim() || "(empty output)", usage: { prompt_tokens: ids.length, completion_tokens: outTok, credits_used: Math.min(actual, reserve), credits_remaining: remaining } };
}
