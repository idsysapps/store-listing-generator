{{/*
Expand the name of the chart.
*/}}
{{- define "store-listing.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "store-listing.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name for postgres.
*/}}
{{- define "store-listing.postgres.fullname" -}}
{{- printf "%s-postgres" (include "store-listing.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name for redis.
*/}}
{{- define "store-listing.redis.fullname" -}}
{{- printf "%s-redis" (include "store-listing.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name for app.
*/}}
{{- define "store-listing.app.fullname" -}}
{{- printf "%s-app" (include "store-listing.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name for trend-service.
*/}}
{{- define "store-listing.trend-service.fullname" -}}
{{- printf "%s-trend-service" (include "store-listing.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "store-listing.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: {{ .Chart.Name }}
{{- end }}

{{/*
Selector labels for app.
*/}}
{{- define "store-listing.app.selectorLabels" -}}
app.kubernetes.io/name: {{ include "store-listing.app.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Selector labels for trend-service.
*/}}
{{- define "store-listing.trend-service.selectorLabels" -}}
app.kubernetes.io/name: {{ include "store-listing.trend-service.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
