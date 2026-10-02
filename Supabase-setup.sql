-- Nova Deploy v2: subscriptions, credits, API keys, leaderboard, server-side execution.
-- Run once in Supabase → SQL Editor. WARNING: replaces the v1 tables (test data is dropped; delete old files in the "models" bucket).
drop table if exists payments, usage, licenses, plans, app_config, models, profiles cascade;
drop function if exists is_active(text), activate_test(text,text), deactivate(text), public_model(text), use_reply(text), grant_license(text,text,text,numeric);
drop policy if exists models_owner_rw on storage.objects; drop policy if exists models_active_read on storage.objects;

create table app_config(key text primary key, value text not null);
insert into app_config values('price_cents','1000'),('currency','usd'),('interval','month'),('credits_per_period','6000000'),
  ('credits_per_input_token','1'),('credits_per_output_token','1');   -- interval: day|week|month|year

create table profiles(id uuid primary key references auth.users on delete cascade,
  display_name text not null check(length(display_name) between 2 and 40), stripe_customer_id text unique);
create table subscriptions(user_id uuid primary key references auth.users on delete cascade,
  status text not null default 'active' check(status in('active','past_due')),
  stripe_subscription_id text, current_period_start timestamptz, current_period_end timestamptz,
  auto_renew boolean not null default true, updated_at timestamptz not null default now());
create table payments(id bigserial primary key, user_id uuid not null references auth.users on delete cascade,
  stripe_subscription_id text, stripe_invoice_id text not null, stripe_payment_intent text,
  amount_cents int not null, currency text not null, status text not null check(status in('paid','failed')),
  period_start timestamptz, period_end timestamptz, is_renewal boolean not null default false,
  renewal_status text, cancellation_status text, credits_granted bigint not null default 0, credits_granted_at timestamptz,
  created timestamptz not null default now(), unique(stripe_invoice_id,status));
create table credit_allocations(id bigserial primary key, user_id uuid not null references auth.users on delete cascade,
  payment_id bigint references payments(id), period_start timestamptz not null, period_end timestamptz not null,
  granted bigint not null check(granted>=0), used bigint not null default 0, granted_at timestamptz not null default now(),
  constraint credits_never_negative check(used>=0 and used<=granted));   -- the database itself refuses negative or overspent credits
create index on credit_allocations(user_id,period_start desc);
create table models(id text primary key, owner uuid not null references auth.users on delete cascade, name text not null,
  version text, created timestamptz not null default now(), source jsonb not null default '{}', sha text not null,
  settings jsonb not null default '{}', deployed boolean not null default true, listed boolean not null default true);
create table api_keys(id uuid primary key default gen_random_uuid(), user_id uuid not null references auth.users on delete cascade,
  name text not null, prefix text not null, key_hash text not null unique, created timestamptz not null default now(),
  last_used timestamptz, revoked_at timestamptz);
create table usage_events(id bigserial primary key, user_id uuid not null references auth.users on delete cascade,
  model_id text references models(id) on delete cascade, via text not null check(via in('playground','api')), api_key_id uuid,
  prompt_tokens int not null, output_tokens int not null, credits bigint not null, ms int, created timestamptz not null default now());
create index on usage_events(model_id); create index on usage_events(user_id,created desc);

alter table app_config enable row level security; alter table profiles enable row level security; alter table subscriptions enable row level security;
alter table payments enable row level security; alter table credit_allocations enable row level security; alter table models enable row level security;
alter table api_keys enable row level security; alter table usage_events enable row level security;
create policy cfg_read on app_config for select using(true);
create policy prof_sel on profiles for select using(id=auth.uid());
create policy prof_ins on profiles for insert with check(id=auth.uid());
create policy prof_upd on profiles for update using(id=auth.uid()) with check(id=auth.uid());
create policy sub_sel on subscriptions for select using(user_id=auth.uid());
create policy pay_sel on payments for select using(user_id=auth.uid());
create policy cred_sel on credit_allocations for select using(user_id=auth.uid());
create policy mod_sel on models for select using(owner=auth.uid());
create policy key_sel on api_keys for select using(user_id=auth.uid());
create policy use_sel on usage_events for select using(user_id=auth.uid());
-- No client insert/update/delete on anything sensitive: models, keys, subscriptions, payments, credits and usage are written only by server code.
revoke all on profiles,subscriptions,payments,credit_allocations,models,api_keys,usage_events from anon,authenticated;
grant select(id,display_name) on profiles to authenticated; grant insert(id,display_name), update(display_name) on profiles to authenticated;
grant select on subscriptions,payments,credit_allocations,models,usage_events to authenticated;
grant select(id,user_id,name,prefix,created,last_used,revoked_at) on api_keys to authenticated;   -- key hashes are never readable by clients
grant select on app_config to anon,authenticated;

create or replace function has_access(p uuid) returns boolean language sql stable security definer set search_path=public as
$$ select exists(select 1 from subscriptions where user_id=p and current_period_end>now()) $$;

-- Single source of truth for the dashboard: status is computed from the paid period, so expiry never depends on a webhook arriving.
create or replace function my_account() returns jsonb language plpgsql stable security definer set search_path=public as $$
declare s subscriptions%rowtype; a credit_allocations%rowtype; st text;
begin
  if auth.uid() is null then return null; end if;
  select * into s from subscriptions where user_id=auth.uid();
  select * into a from credit_allocations where user_id=auth.uid() and period_start<=now() and period_end>now() order by period_start desc,id desc limit 1;
  st:=case when s.user_id is null or s.current_period_end is null then 'none' when s.current_period_end<=now() then 'expired'
           when s.status='past_due' then 'past_due' when not s.auto_renew then 'cancelled' else 'active' end;
  if st not in('active','past_due','cancelled') then a:=null; end if;   -- no credits are usable without a paid period
  return jsonb_build_object('status',st,'access',st in('active','past_due','cancelled'),'period_start',s.current_period_start,
    'period_end',s.current_period_end,'auto_renew',coalesce(s.auto_renew,false),'granted',coalesce(a.granted,0),'used',coalesce(a.used,0),
    'remaining',coalesce(a.granted-a.used,0),'has_stripe_sub',s.stripe_subscription_id is not null,
    'price_cents',(select value::int from app_config where key='price_cents'),'currency',(select value from app_config where key='currency'),
    'interval',(select value from app_config where key='interval'),'credits_per_period',(select value::bigint from app_config where key='credits_per_period'));
end $$;

-- Public, read-only leaderboard (no weights, no way to run a model).
create or replace function leaderboard(p_limit int default 50) returns table(rank bigint,model_id text,name text,creator text,version text,
  replies bigint,avg_ms int,steps int,epochs int,params bigint) language sql stable security definer set search_path=public as $$
  select row_number() over(order by coalesce(u.replies,0) desc,(m.source->>'steps')::int desc nulls last,m.created),
    m.id,m.name,p.display_name,m.version,coalesce(u.replies,0),u.avg_ms::int,(m.source->>'steps')::int,(m.source->>'epochs')::int,(m.source->>'params')::bigint
  from models m join profiles p on p.id=m.owner
  left join(select model_id,count(*) replies,avg(ms) avg_ms from usage_events group by 1) u on u.model_id=m.id
  where m.deployed and m.listed order by 1 limit least(greatest(p_limit,1),100) $$;
create or replace function public_model_info(p_id text) returns jsonb language sql stable security definer set search_path=public as $$
  select jsonb_build_object('id',m.id,'name',m.name,'creator',p.display_name,'version',m.version,'description',m.settings->>'desc','created',m.created,
    'replies',(select count(*) from usage_events e where e.model_id=m.id),'steps',m.source->'steps','epochs',m.source->'epochs','params',m.source->'params')
  from models m join profiles p on p.id=m.owner where m.id=p_id and m.deployed and m.listed $$;

-- ===== Server-only (service role): payments, credits, usage =====
create or replace function user_for_customer(p_cust text) returns uuid language sql stable security definer set search_path=public as
$$ select id from profiles where stripe_customer_id=p_cust $$;

-- Called only after Stripe confirms an invoice is PAID. Idempotent per invoice (Stripe may resend events).
create or replace function grant_period(p_user uuid,p_invoice text,p_pi text,p_sub text,p_cents int,p_cur text,p_start timestamptz,p_end timestamptz,p_renewal boolean)
returns void language plpgsql security definer set search_path=public as $$
declare cr bigint; pid bigint; old subscriptions%rowtype;
begin
  if exists(select 1 from payments where stripe_invoice_id=p_invoice and status='paid') then return; end if;
  select value::bigint into cr from app_config where key='credits_per_period';
  select * into old from subscriptions where user_id=p_user;
  insert into payments(user_id,stripe_subscription_id,stripe_invoice_id,stripe_payment_intent,amount_cents,currency,status,period_start,period_end,is_renewal,renewal_status,cancellation_status,credits_granted,credits_granted_at)
    values(p_user,p_sub,p_invoice,p_pi,p_cents,p_cur,'paid',p_start,p_end,p_renewal,case when p_renewal then 'renewed' else 'new' end,'auto_renew_on',cr,now()) returning id into pid;
  insert into credit_allocations(user_id,payment_id,period_start,period_end,granted) values(p_user,pid,p_start,p_end,cr);   -- fresh allocation; nothing carries over
  if old.user_id is null then
    insert into subscriptions(user_id,status,stripe_subscription_id,current_period_start,current_period_end,auto_renew) values(p_user,'active',p_sub,p_start,p_end,true);
  elsif old.stripe_subscription_id is distinct from p_sub then   -- new subscription (e.g. resubscribe after expiry)
    update subscriptions set status='active',stripe_subscription_id=p_sub,current_period_start=p_start,current_period_end=p_end,auto_renew=true,updated_at=now() where user_id=p_user;
  elsif p_end>=old.current_period_end then
    update subscriptions set status='active',current_period_start=p_start,current_period_end=p_end,updated_at=now() where user_id=p_user;
  end if;
end $$;

create or replace function record_failed_payment(p_user uuid,p_invoice text,p_pi text,p_sub text,p_cents int,p_cur text) returns void language plpgsql security definer set search_path=public as $$
begin   -- never grants credits, never extends the period
  insert into payments(user_id,stripe_subscription_id,stripe_invoice_id,stripe_payment_intent,amount_cents,currency,status,renewal_status,cancellation_status)
    values(p_user,p_sub,p_invoice,p_pi,p_cents,p_cur,'failed','payment_failed','auto_renew_on') on conflict do nothing;
  update subscriptions set status='past_due',updated_at=now() where user_id=p_user and stripe_subscription_id=p_sub;
end $$;

create or replace function sync_renewal(p_sub text,p_auto boolean) returns void language sql security definer set search_path=public as
$$ update subscriptions set auto_renew=p_auto,updated_at=now() where stripe_subscription_id=p_sub $$;

-- Atomic reservation: the UPDATE row-locks the allocation, so concurrent requests can never spend the same credits twice.
create or replace function reserve_credits(p_user uuid,p_amount bigint) returns bigint language plpgsql security definer set search_path=public as $$
declare aid bigint;
begin
  if p_amount<=0 or not has_access(p_user) then return null; end if;
  update credit_allocations set used=used+p_amount
   where id=(select id from credit_allocations where user_id=p_user and period_start<=now() and period_end>now() order by period_start desc,id desc limit 1)
     and used+p_amount<=granted returning id into aid;
  return aid;
end $$;
create or replace function release_credits(p_alloc bigint,p_amount bigint) returns void language sql security definer set search_path=public as
$$ update credit_allocations set used=greatest(0,used-p_amount) where id=p_alloc $$;
create or replace function settle_credits(p_alloc bigint,p_user uuid,p_reserved bigint,p_actual bigint,p_model text,p_via text,p_key uuid,p_in int,p_out int,p_ms int)
returns bigint language plpgsql security definer set search_path=public as $$
declare c bigint:=least(p_actual,p_reserved); rem bigint;
begin
  update credit_allocations set used=used-(p_reserved-c) where id=p_alloc and user_id=p_user returning granted-used into rem;
  insert into usage_events(user_id,model_id,via,api_key_id,prompt_tokens,output_tokens,credits,ms) values(p_user,p_model,p_via,p_key,p_in,p_out,c,p_ms);
  return rem;
end $$;

revoke all on function has_access(uuid),my_account(),leaderboard(int),public_model_info(text),user_for_customer(text),
  grant_period(uuid,text,text,text,int,text,timestamptz,timestamptz,boolean),record_failed_payment(uuid,text,text,text,int,text),sync_renewal(text,boolean),
  reserve_credits(uuid,bigint),release_credits(bigint,bigint),settle_credits(bigint,uuid,bigint,bigint,text,text,uuid,int,int,int) from public,anon,authenticated;
grant execute on function my_account() to authenticated;
grant execute on function leaderboard(int),public_model_info(text) to anon,authenticated;
grant execute on function has_access(uuid),user_for_customer(text),grant_period(uuid,text,text,text,int,text,timestamptz,timestamptz,boolean),
  record_failed_payment(uuid,text,text,text,int,text),sync_renewal(text,boolean),reserve_credits(uuid,bigint),release_credits(bigint,bigint),
  settle_credits(bigint,uuid,bigint,bigint,text,text,uuid,int,int,int) to service_role;

-- Weights live in a private bucket that only server code can read. Clients can upload only into their own folder, and only with an active subscription.
insert into storage.buckets(id,name,public) values('models','models',false) on conflict do nothing;
create policy models_owner_upload on storage.objects for insert to authenticated
  with check(bucket_id='models' and (storage.foldername(name))[1]=auth.uid()::text and has_access(auth.uid()));
grant execute on function has_access(uuid) to authenticated;
