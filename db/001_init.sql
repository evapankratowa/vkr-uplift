-- Модель СУБД PostgreSQL 16: операционная схема uplift и звезда uplift_dw.
-- Получена из VKR_ch2_11_transform.puml и VKR_ch2_12_dimensional.puml.
-- Проверка: psql -f VKR_ch2_13_dbms.sql на пустой БД.

create schema if not exists uplift;
create schema if not exists uplift_dw;
set search_path to uplift;

create type cashback_source as enum ('partner', 'bank');
create type experiment_source as enum ('stand', 'real');
create type model_type as enum ('response_baseline', 'automl');
create type model_status as enum ('trained', 'evaluated', 'accepted', 'rejected', 'in_use', 'retired');
create type request_status as enum ('created', 'checking', 'feasible', 'infeasible', 'corrected', 'cancelled', 'segment_formed', 'exported');
create type export_format as enum ('standard', 'ab');
create type strategy_group as enum ('model_promo', 'random_promo');

create table client
(
client_id bigint primary key
);

create table spend_category
(
category_id smallint primary key
, name varchar(32) not null unique
);

create table observation_window
(
window_id smallint primary key
, code varchar(16) not null unique
, date_from date
, date_to date
, check (date_to is null or date_from is null or date_to >= date_from)
);

create table client_spend
(
client_id bigint not null references client (client_id)
, category_id smallint not null references spend_category (category_id)
, window_id smallint not null references observation_window (window_id)
, amount numeric(14, 2) not null check (amount >= 0)
, primary key (client_id, category_id, window_id)
);

-- Отрицательные значения разрешены: сторно кэшбэка при возврате покупки.
create table client_cashback
(
client_id bigint not null references client (client_id)
, window_id smallint not null references observation_window (window_id)
, source cashback_source not null
, amount numeric(14, 2) not null
, primary key (client_id, window_id, source)
);

create table communication_stats
(
client_id bigint primary key references client (client_id)
, window_id smallint not null references observation_window (window_id)
, cnt_communication integer not null check (cnt_communication >= 0)
, cnt_view integer not null check (cnt_view >= 0)
, cnt_activation integer not null check (cnt_activation >= 0)
);

-- В прототипе — дамми-справочник; лимит означает один контакт не чаще N дней.
create table contact_consent
(
client_id bigint not null references client (client_id)
, channel varchar(16) not null
, consent boolean not null
, min_interval_days smallint check (min_interval_days >= 0)
, last_contact_date date
, primary key (client_id, channel)
);

create table ab_experiment
(
experiment_id serial primary key
, source experiment_source not null
, control_share numeric(4, 3) not null check (control_share > 0 and control_share < 1)
, start_date date
, end_date date
, check (end_date is null or start_date is null or end_date >= start_date)
);

-- treatment означает факт промо / отсутствие промо, а не стратегию отбора.
create table ab_assignment
(
experiment_id integer not null references ab_experiment (experiment_id)
, client_id bigint not null references client (client_id)
, treatment boolean not null
, target boolean not null
, primary key (experiment_id, client_id)
);
create index ix_ab_assignment_client on ab_assignment (client_id);

create table scoring_model
(
model_id serial primary key
, experiment_id integer not null references ab_experiment (experiment_id)
, model_type model_type not null
, meta_learner varchar(32)
, base_algorithm varchar(32)
, hyperparams jsonb not null default '{}'::jsonb
, status model_status not null default 'trained'
, trained_at timestamptz not null default now()
, artifact_uri text not null
, training_config jsonb not null default '{}'::jsonb
, feature_config jsonb not null default '{}'::jsonb
, check (
    (model_type = 'automl' and meta_learner is not null)
    or (model_type = 'response_baseline' and meta_learner is null)
)
);
create index ix_scoring_model_experiment on scoring_model (experiment_id);

create table model_metric
(
model_id integer not null references scoring_model (model_id)
, metric varchar(32) not null
, value double precision not null
, primary key (model_id, metric)
);

create table score_run
(
score_run_id bigserial primary key
, model_id integer not null references scoring_model (model_id)
, scored_at timestamptz not null default now()
, input_snapshot text
, parameters jsonb not null default '{}'::jsonb
);
create index ix_score_run_model on score_run (model_id);

create table client_score_history
(
score_run_id bigint not null references score_run (score_run_id)
, client_id bigint not null references client (client_id)
, uplift_score double precision not null
, primary key (score_run_id, client_id)
);
create index ix_client_score_history_rank on client_score_history (score_run_id, uplift_score desc);

create table segment_request
(
request_id serial primary key
, parent_request_id integer references segment_request (request_id)
, model_id integer not null references scoring_model (model_id)
, score_run_id bigint references score_run (score_run_id)
, channel varchar(16) not null
, target_reach integer not null check (target_reach > 0)
, target_fin_result numeric(14, 2) not null check (target_fin_result >= 0)
, margin numeric(10, 2) not null check (margin >= 0)
, contact_cost numeric(10, 2) not null check (contact_cost >= 0)
, status request_status not null default 'created'
, created_at timestamptz not null default now()
, check (parent_request_id is null or parent_request_id <> request_id)
);
create index ix_segment_request_parent on segment_request (parent_request_id);
create index ix_segment_request_model on segment_request (model_id);

create table feasibility_response
(
request_id integer primary key references segment_request (request_id)
, feasible boolean not null
, available_client_count integer not null check (available_client_count >= 0)
, max_fin_result_at_target_reach numeric(14, 2)
, max_reach_at_target_fin_result integer not null check (max_reach_at_target_fin_result >= 0)
, selected_segment_size integer check (selected_segment_size > 0)
, check (
    (feasible and selected_segment_size is not null)
    or (not feasible and selected_segment_size is null)
)
);

create table segment
(
segment_id serial primary key
, request_id integer not null unique references segment_request (request_id)
, size integer not null check (size > 0)
, expected_fin_result numeric(14, 2) not null
);

create table segment_member
(
segment_id integer not null references segment (segment_id)
, client_id bigint not null references client (client_id)
, uplift_score double precision not null
, primary key (segment_id, client_id)
);

create table segment_export
(
export_id bigserial primary key
, segment_id integer not null references segment (segment_id)
, export_format export_format not null
, file_uri text not null
, random_seed bigint
, exported_at timestamptz not null default now()
, check (
    (export_format = 'standard' and random_seed is null)
    or (export_format = 'ab' and random_seed is not null)
)
);
create index ix_segment_export_segment on segment_export (segment_id);

-- PK не допускает пересечения model_promo и random_promo в одной выгрузке.
create table segment_export_member
(
export_id bigint not null references segment_export (export_id)
, client_id bigint not null references client (client_id)
, strategy_group strategy_group not null
, uplift_score double precision not null
, primary key (export_id, client_id)
);

create table contact_event
(
contact_id bigserial primary key
, export_id bigint not null
, client_id bigint not null
, channel varchar(16) not null
, contacted_at timestamptz not null
, foreign key (export_id, client_id)
    references segment_export_member (export_id, client_id)
);
create index ix_contact_event_export_client on contact_event (export_id, client_id);

-- Размерное аналитическое хранилище.
set search_path to uplift_dw;

create table dim_client
(
client_key bigint primary key
, client_id bigint not null unique
);

create table dim_campaign
(
campaign_key serial primary key
, request_id integer not null unique
, segment_id integer not null unique
, target_reach integer not null
, target_fin_result numeric(14, 2) not null
, margin numeric(10, 2) not null
, contact_cost numeric(10, 2) not null
, feasible boolean not null
);

create table dim_model
(
model_key serial primary key
, model_id integer not null unique
, model_type varchar(32) not null
, meta_learner varchar(32)
, base_algorithm varchar(32)
, qini double precision
);

create table dim_date
(
date_key integer primary key
, full_date date not null unique
, year smallint not null
, month smallint not null
, week smallint not null
);

create table dim_channel
(
channel_key smallserial primary key
, channel varchar(16) not null unique
);

create table dim_category
(
category_key smallint primary key
, name varchar(32) not null
);

create table dim_window
(
window_key smallint primary key
, code varchar(16) not null
, date_from date
, date_to date
);

create table dim_score_run
(
score_run_key bigint primary key
, score_run_id bigint not null unique
, model_key integer not null references dim_model (model_key)
, scored_at timestamptz not null
, input_snapshot text
);

-- Гранулярность: один фактический контакт, полученный из CRM.
create table fact_segment_contact
(
contact_id bigint primary key
, client_key bigint not null references dim_client (client_key)
, campaign_key integer not null references dim_campaign (campaign_key)
, model_key integer not null references dim_model (model_key)
, date_key integer not null references dim_date (date_key)
, channel_key smallint not null references dim_channel (channel_key)
, export_id bigint not null
, strategy_group varchar(16) not null check (strategy_group in ('model_promo', 'random_promo'))
, uplift_score double precision
, expected_contact_cost numeric(10, 2)
, expected_incremental_profit numeric(14, 2)
);

create table fact_client_spend
(
client_key bigint not null references dim_client (client_key)
, category_key smallint not null references dim_category (category_key)
, window_key smallint not null references dim_window (window_key)
, spend_amount numeric(14, 2) not null
, primary key (client_key, category_key, window_key)
);

-- Гранулярность: клиент × запуск скоринга.
create table fact_client_score_history
(
client_key bigint not null references dim_client (client_key)
, score_run_key bigint not null references dim_score_run (score_run_key)
, model_key integer not null references dim_model (model_key)
, date_key integer not null references dim_date (date_key)
, uplift_score double precision not null
, primary key (client_key, score_run_key)
);
