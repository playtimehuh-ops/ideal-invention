// Authenticated app backend (JWT required). Every action re-checks identity, subscription and ownership on the server.
// Secrets: STRIPE_SECRET_KEY, SITE_URL (exact https URL of the page)
import Stripe from "npm:stripe@17";
import { createClient } from "npm:@supabase/supabase-js@2";
import { admin, buildModel, config, HttpErr, runChat, sha256 } from "../_shared/run.ts";

const stripe = new Stripe(Deno.env.get("STRIPE_SECRET_KEY")!, { httpClient: Stripe.createFetchHttpClient() });
const cors = { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type" };
const json = (b: unknown, s = 200) => new Response(JSON.stringify(b), { status: s, headers: { ...cors, "Content-Type": "application/json" } });
const sub402 = () => new HttpErr(402, "subscription_required", "An active subscription is required.");

async function requireAccess(uid: string) { const { data } = await admin.rpc("has_access", { p: uid }); if (!data) throw sub402(); }
async function ownModel(uid: string, id: string) {
  const { data } = await admin.from("models").select("*").eq("id", String(id)).maybeSingle();
  if (!data || data.owner !== uid) throw new HttpErr(404, "model_not_found", "Model not found.");
  return data;
}
const cleanSettings = (s: any) => ({
  desc: String(s?.desc ?? "").slice(0, 140), temp: Math.min(2, Math.max(0.1, Number(s?.temp) || 0.8)), topK: Math.min(200, Math.max(1, Math.round(Number(s?.topK) || 30))),
  topP: Math.min(1, Math.max(0.1, Number(s?.topP) || 0.9)), maxTok: Math.min(300, Math.max(8, Math.round(Number(s?.maxTok) || 80))),
});

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: cors });
  try {
    const userClient = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_ANON_KEY")!, { global: { headers: { Authorization: req.headers.get("Authorization") ?? "" } } });
    const { data: { user } } = await userClient.auth.getUser();
    if (!user) throw new HttpErr(401, "unauthenticated", "Sign in first.");
    const b = await req.json(); const uid = user.id;

    switch (b.action) {
      case "register_model": {   // create + deploy: needs a subscription
        await requireAccess(uid);
        const id = String(b.id ?? ""); if (!/^mdl_[0-9a-f]{10}$/.test(id)) throw new HttpErr(400, "bad_id", "Bad model id.");
        const name = String(b.name ?? "").trim().slice(0, 40); if (!name) throw new HttpErr(400, "bad_name", "Give your AI a name.");
        const path = `${uid}/${id}.json`;
        try {
          const { data: file, error } = await admin.storage.from("models").download(path);
          if (error || !file) throw new HttpErr(400, "no_upload", "Upload the model file first.");
          const text = await file.text(), o = JSON.parse(text);
          const { m, t } = buildModel(o);
          m.generate(t.encode("User: hi\nNova:"), 2, 0.8, 30, 0.9, { stop: false });          // smoke test with the real engine
          const { error: e2 } = await admin.from("models").insert({ id, owner: uid, name, version: "1.0." + (o.stepCount || 0), sha: await sha256(text),
            settings: cleanSettings(b.settings), source: { steps: o.stepCount || 0, epochs: o.epochsDone || 0, params: m.paramCount(), vocab: o.vocabSize, config: o.config, exportedAt: o.createdAt ?? null } });
          if (e2) throw new HttpErr(500, "db", e2.message);
        } catch (e) { await admin.storage.from("models").remove([path]); throw e; }
        return json({ id });
      }
      case "update_model": {
        const m = await ownModel(uid, b.id), patch: Record<string, unknown> = {};
        if (typeof b.name === "string" && b.name.trim()) patch.name = b.name.trim().slice(0, 40);
        if (b.settings) patch.settings = cleanSettings(b.settings);
        if (typeof b.listed === "boolean") patch.listed = b.listed;
        if (typeof b.deployed === "boolean") { if (b.deployed) await requireAccess(uid); patch.deployed = b.deployed; }
        if (Object.keys(patch).length) await admin.from("models").update(patch).eq("id", m.id);
        return json({ ok: true });
      }
      case "delete_model": {
        const m = await ownModel(uid, b.id);
        await admin.storage.from("models").remove([`${uid}/${m.id}.json`]); await admin.from("models").delete().eq("id", m.id);
        return json({ ok: true });
      }
      case "chat":   // playground: runs the real model on the server, same credits as the API
        return json(await runChat({ userId: uid, modelId: String(b.modelId ?? ""), messages: b.messages, via: "playground" }));
      case "create_api_key": {
        await requireAccess(uid);
        const { count } = await admin.from("api_keys").select("id", { count: "exact", head: true }).eq("user_id", uid).is("revoked_at", null);
        if ((count ?? 0) >= 10) throw new HttpErr(400, "too_many_keys", "Revoke an old key first (max 10 active).");
        const key = "nova_sk_" + btoa(String.fromCharCode(...crypto.getRandomValues(new Uint8Array(32)))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
        const { error } = await admin.from("api_keys").insert({ user_id: uid, name: String(b.name ?? "").trim().slice(0, 40) || "Untitled key", prefix: key.slice(0, 16), key_hash: await sha256(key) });
        if (error) throw new HttpErr(500, "db", error.message);
        return json({ key });   // the only time the secret is ever shown; only its hash is stored
      }
      case "revoke_api_key":
        await admin.from("api_keys").update({ revoked_at: new Date().toISOString() }).eq("id", String(b.id)).eq("user_id", uid).is("revoked_at", null);
        return json({ ok: true });
      case "checkout": {   // Stripe Checkout in subscription mode; price and credits come from app_config, never from the browser
        const { data: s } = await admin.from("subscriptions").select("*").eq("user_id", uid).maybeSingle();
        if (s?.current_period_end && new Date(s.current_period_end) > new Date())
          throw new HttpErr(409, "already_subscribed", s.auto_renew ? "You already have an active subscription." : "Your paid period is still running. Resume auto-renew instead.");
        const { data: p } = await admin.from("profiles").select("stripe_customer_id").eq("id", uid).maybeSingle();
        if (!p) throw new HttpErr(400, "no_profile", "Finish creating your profile first.");
        let cust = p.stripe_customer_id;
        if (!cust) { cust = (await stripe.customers.create({ email: user.email, metadata: { user_id: uid } })).id; await admin.from("profiles").update({ stripe_customer_id: cust }).eq("id", uid); }
        const cfg = await config(), ok = new URL(Deno.env.get("SITE_URL")!), no = new URL(Deno.env.get("SITE_URL")!);
        ok.searchParams.set("paid", "1"); no.searchParams.set("canceled", "1");
        const session = await stripe.checkout.sessions.create({
          mode: "subscription", customer: cust, client_reference_id: uid, success_url: ok.toString(), cancel_url: no.toString(),
          line_items: [{ quantity: 1, price_data: { currency: cfg.currency, unit_amount: Number(cfg.price_cents), recurring: { interval: cfg.interval as "month" },
            product_data: { name: `Nova Deploy: ${Number(cfg.credits_per_period).toLocaleString()} credits per ${cfg.interval}` } } }],
        });
        return json({ url: session.url });
      }
      case "set_renewal": {   // cancel = stop charging after the paid period; access is untouched until it ends
        const { data: s } = await admin.from("subscriptions").select("*").eq("user_id", uid).maybeSingle();
        if (!s?.stripe_subscription_id || new Date(s.current_period_end) <= new Date()) throw new HttpErr(400, "no_active_subscription", "No active subscription to change.");
        await stripe.subscriptions.update(s.stripe_subscription_id, { cancel_at_period_end: !b.on });
        await admin.rpc("sync_renewal", { p_sub: s.stripe_subscription_id, p_auto: !!b.on });
        return json({ ok: true });
      }
      default: throw new HttpErr(400, "unknown_action", "Unknown action.");
    }
  } catch (e) {
    if (e instanceof HttpErr) return json({ error: e.message, code: e.code }, e.status);
    console.error(e); return json({ error: "Internal error", code: "server_error" }, 500);
  }
});
