output "agent_service_account_email" {
  description = "Email of the least-privilege service account used by the ADK agent runtime."
  value       = google_service_account.studyops_agent_sa.email
}

output "firestore_database_name" {
  description = "Cloud Firestore database resource name storing learner SRS flashcards."
  value       = google_firestore_database.learner_srs_store.name
}

output "vertex_search_data_store_id" {
  description = "Vertex AI Search (Discovery Engine) data store ID for JLPT grammar retrieval."
  value       = google_discovery_engine_data_store.jlpt_grammar_corpus.data_store_id
}

output "secret_manager_secret_id" {
  description = "Secret Manager secret identifier for runtime configuration."
  value       = google_secret_manager_secret.anki_sync_api_secret.secret_id
}
