// Public API:  POST /functions/v1/v1   Authorization: Bearer nova_sk_...
// body: { "model": "mdl_...", "messages": [{ "role": "user", "content": "Hello" }], "max_tokens": 80, "temperature": 0.8 }
// Deploy WITHOUT gateway JWT checking (keys are validated here):  supabase functions deploy v1 --no-verify-jwt
import { admin, HttpErr, runChat, sha256 } from "../_shared/run.ts";

const json = (b: unknown, s = 200) => new Response(JSON.stringify(b), { status: s, headers: { "Content-Type": "application/json" } });
const fail = (e: unknown) => e instanceof HttpErr ? json({ error: { code: e.code, message: e.message } }, e.status) : json({ error: { code: "server_error", message: "Internal error" } }, 500);

Deno.serve(async (req) => {
  try {
    if (req.method !== "POST") throw new HttpErr(405, "method_not_allowed", "Use POST.");
    const m = /^Bearer (nova_sk_[A-Za-z0-9_-]{20,})$/.exec(req.headers.get("Authorization") ?? "");
    if (!m) throw new HttpErr(401, "invalid_api_key", "Missing or malformed API key.");
    const { data: key } = await admin.from("api_keys").select("id,user_id,revoked_at").eq("key_hash", await sha256(m[1])).maybeSingle();
    if (!key || key.revoked_at) throw new HttpErr(401, "invalid_api_key", "Invalid or revoked API key.");
    const body = await req.json().catch(() => { throw new HttpErr(400, "bad_json", "Body must be JSON."); });
    // runChat re-checks the live subscription, credits and model ownership on every call, so an expired subscription stops the key instantly.
    const r = await runChat({ userId: key.user_id, modelId: String(body.model ?? ""), messages: body.messages, maxTokens: body.max_tokens, temperature: body.temperature, via: "api", keyId: key.id });
    admin.from("api_keys").update({ last_used: new Date().toISOString() }).eq("id", key.id).then(() => {});
    return json({ id: "cmpl_" + crypto.randomUUID().slice(0, 12), model: body.model, reply: r.reply, usage: r.usage });
  } catch (e) { return fail(e); }
});
