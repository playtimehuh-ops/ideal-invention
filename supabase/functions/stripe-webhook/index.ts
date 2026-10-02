import Stripe from "npm:stripe@17";
import { createClient } from "npm:@supabase/supabase-js@2";

const stripe = new Stripe(Deno.env.get("STRIPE_SECRET_KEY")!, { httpClient: Stripe.createFetchHttpClient() });
const admin = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!, { auth: { persistSession: false } });
const secret = Deno.env.get("STRIPE_WEBHOOK_SECRET")!;
const cors = {"Access-Control-Allow-Origin":"*"};

Deno.serve(async (req) => {
  if (req.method !== "POST") return new Response("Method Not Allowed",{status:405,headers:cors});
  const sig=req.headers.get("stripe-signature");
  if(!sig) return new Response("Missing signature",{status:400,headers:cors});
  const raw=await req.text();
  let event: Stripe.Event;
  try { event=await stripe.webhooks.constructEventAsync(raw,sig,secret); }
  catch(e){ console.error("stripe signature verification failed",e); return new Response("Invalid signature",{status:400,headers:cors}); }
  try {
    if(event.type==="invoice.paid"){
      const inv=event.data.object as Stripe.Invoice;
      const subId=typeof inv.subscription==="string"?inv.subscription:inv.subscription?.id;
      if(!subId) throw new Error("invoice has no subscription");
      const sub=await stripe.subscriptions.retrieve(subId);
      const userId=String(sub.metadata?.user_id || inv.metadata?.user_id || "");
      let uid=userId;
      if(!uid && inv.customer){
        const {data}=await admin.rpc("deploy_user_for_customer",{p_cust:String(inv.customer)}); uid=data || "";
      }
      if(!uid) throw new Error("Cannot resolve subscription owner");
      await admin.rpc("deploy_grant_period",{p_user:uid,p_invoice:inv.id,p_pi:typeof inv.payment_intent==="string"?inv.payment_intent:null,p_sub:sub.id,
        p_cents:inv.amount_paid,p_cur:inv.currency,p_start:new Date(sub.current_period_start*1000).toISOString(),
        p_end:new Date(sub.current_period_end*1000).toISOString(),p_renewal:(inv.billing_reason==="subscription_cycle")});
    } else if(event.type==="invoice.payment_failed"){
      const inv=event.data.object as Stripe.Invoice;
      const subId=typeof inv.subscription==="string"?inv.subscription:inv.subscription?.id;
      if(subId){
        const sub=await stripe.subscriptions.retrieve(subId);
        const uid=String(sub.metadata?.user_id || inv.metadata?.user_id || "");
        if(uid) await admin.rpc("deploy_record_failed_payment",{p_user:uid,p_invoice:inv.id,p_pi:typeof inv.payment_intent==="string"?inv.payment_intent:null,p_sub:sub.id,p_cents:inv.amount_due,p_cur:inv.currency});
      }
    } else if(event.type==="customer.subscription.updated"){
      const sub=event.data.object as Stripe.Subscription;
      await admin.rpc("deploy_sync_renewal",{p_sub:sub.id,p_auto:!sub.cancel_at_period_end});
    } else if(event.type==="customer.subscription.deleted"){
      const sub=event.data.object as Stripe.Subscription;
      await admin.rpc("deploy_sync_renewal",{p_sub:sub.id,p_auto:false});
    }
    return new Response(JSON.stringify({received:true}),{status:200,headers:{...cors,"Content-Type":"application/json"}});
  } catch(e){ console.error("webhook processing failed",e); return new Response("Webhook processing failed",{status:500,headers:cors}); }
});