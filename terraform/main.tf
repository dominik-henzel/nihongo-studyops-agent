terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.30"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  required_apis = toset([
    "aiplatform.googleapis.com",
    "discoveryengine.googleapis.com",
    "firestore.googleapis.com",
    "dlp.googleapis.com",
    "cloudtrace.googleapis.com",
    "logging.googleapis.com",
    "secretmanager.googleapis.com",
  ])

  agent_iam_roles = toset([
    "roles/aiplatform.user",
    "roles/discoveryengine.viewer",
    "roles/datastore.user",
    "roles/dlp.user",
    "roles/cloudtrace.agent",
    "roles/logging.logWriter",
    "roles/secretmanager.secretAccessor",
  ])
}

# 1. Enable required Google Cloud APIs for ADK runtime, memory, DLP, and tracing.
resource "google_project_service" "required_services" {
  for_each           = local.required_apis
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# 2. Dedicated least-privilege Service Account for the Nihongo StudyOps Agent.
resource "google_service_account" "studyops_agent_sa" {
  account_id   = "nihongo-studyops-agent-${var.environment}"
  display_name = "Nihongo StudyOps ADK Agent Service Account (${var.environment})"
  description  = "Least-privilege identity for the Nihongo StudyOps multi-agent pipeline."
}

resource "google_project_iam_member" "studyops_agent_roles" {
  for_each = local.agent_iam_roles
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.studyops_agent_sa.email}"
}

# 3. Cloud Firestore Native Database for persistent learner vocabulary & SM-2 SRS decks.
resource "google_firestore_database" "learner_srs_store" {
  project                     = var.project_id
  name                        = "nihongo-studyops-srs-${var.environment}"
  location_id                 = var.firestore_location
  type                        = "FIRESTORE_NATIVE"
  concurrency_mode            = "OPTIMISTIC"
  app_engine_integration_mode = "DISABLED"

  depends_on = [google_project_service.required_services]
}

# 4. Vertex AI Search (Discovery Engine) Data Store for the JLPT N5-N1 grammar corpus.
resource "google_discovery_engine_data_store" "jlpt_grammar_corpus" {
  project                     = var.project_id
  location                    = var.discovery_engine_location
  data_store_id               = "nihongo-jlpt-grammar-corpus-${var.environment}"
  display_name                = "JLPT N5-N1 Grammar Corpus (${var.environment})"
  industry_vertical           = "GENERIC"
  content_config              = "NO_CONTENT"
  solution_types              = ["SOLUTION_TYPE_SEARCH"]
  create_advanced_site_search = false

  depends_on = [google_project_service.required_services]
}

# 5. Google Cloud Secret Manager secret container for external dictionary / Anki sync keys.
resource "google_secret_manager_secret" "anki_sync_api_secret" {
  project   = var.project_id
  secret_id = "nihongo-studyops-anki-sync-token-${var.environment}"

  labels = {
    app         = "nihongo-studyops-agent"
    environment = var.environment
  }

  replication {
    auto {}
  }

  depends_on = [google_project_service.required_services]
}
