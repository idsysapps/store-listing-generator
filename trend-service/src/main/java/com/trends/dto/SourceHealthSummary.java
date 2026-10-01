package com.trends.dto;

import java.time.OffsetDateTime;

public class SourceHealthSummary {

    private String source;
    private int consecutiveFailures;
    private int consecutiveEmpty;
    private String lastError;
    private OffsetDateTime lastErrorAt;
    private OffsetDateTime lastSuccessAt;
    private boolean issueOpen;
    private String lastIssueUrl;

    public SourceHealthSummary() {}

    public SourceHealthSummary(String source, int consecutiveFailures, int consecutiveEmpty,
            String lastError, OffsetDateTime lastErrorAt, OffsetDateTime lastSuccessAt,
            boolean issueOpen, String lastIssueUrl) {
        this.source = source;
        this.consecutiveFailures = consecutiveFailures;
        this.consecutiveEmpty = consecutiveEmpty;
        this.lastError = lastError;
        this.lastErrorAt = lastErrorAt;
        this.lastSuccessAt = lastSuccessAt;
        this.issueOpen = issueOpen;
        this.lastIssueUrl = lastIssueUrl;
    }

    public String getSource() { return source; }
    public int getConsecutiveFailures() { return consecutiveFailures; }
    public int getConsecutiveEmpty() { return consecutiveEmpty; }
    public String getLastError() { return lastError; }
    public OffsetDateTime getLastErrorAt() { return lastErrorAt; }
    public OffsetDateTime getLastSuccessAt() { return lastSuccessAt; }
    public boolean isIssueOpen() { return issueOpen; }
    public String getLastIssueUrl() { return lastIssueUrl; }
}
