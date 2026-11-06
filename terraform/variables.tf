variable "project_id" {
  description = "Google Cloud project ID hosting the Nihongo StudyOps Agent resources."
  type        = string
  default     = "nihongo-studyops-demo"
}

variable "region" {
  description = "Primary Google Cloud region for Vertex AI Agent Engine and Firestore."
  type        = string
  default     = "us-central1"
}

variable "firestore_location" {
  description = "Location ID for the Cloud Firestore Native database storing learner SRS state."
  type        = string
  default     = "nam5"
}

variable "discovery_engine_location" {
  description = "Location for the Vertex AI Search (Discovery Engine) JLPT grammar data store."
  type        = string
  default     = "global"
}

variable "environment" {
  description = "Deployment environment label (e.g., dev, staging, prod)."
  type        = string
  default     = "prod"
}
