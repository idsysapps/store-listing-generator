package com.trends.domain;

import io.quarkus.hibernate.orm.panache.PanacheEntityBase;
import jakarta.persistence.*;
import java.time.OffsetDateTime;

@Entity
@Table(name = "source_health")
public class SourceHealth extends PanacheEntityBase {

    @Id
    @Column(name = "source", length = 20)
    public String source;

    @Column(name = "consecutive_failures", nullable = false)
    public int consecutiveFailures;

    @Column(name = "consecutive_empty", nullable = false)
    public int consecutiveEmpty;

    @Column(name = "last_error", columnDefinition = "TEXT")
    public String lastError;

    @Column(name = "last_error_at")
    public OffsetDateTime lastErrorAt;

    @Column(name = "last_success_at")
    public OffsetDateTime lastSuccessAt;

    @Column(name = "issue_open", nullable = false)
    public boolean issueOpen;

    @Column(name = "last_issue_url", columnDefinition = "TEXT")
    public String lastIssueUrl;

    @Column(name = "last_issue_number")
    public Integer lastIssueNumber;

    @Column(name = "updated_at")
    public OffsetDateTime updatedAt;

    public SourceHealth() {}
}
