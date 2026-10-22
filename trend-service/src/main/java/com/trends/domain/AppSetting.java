package com.trends.domain;

import io.quarkus.hibernate.orm.panache.PanacheEntityBase;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import java.time.OffsetDateTime;

/**
 * A setting the operator can change without a redeploy.
 *
 * <p>The product types the pipeline produces right now are stored here, so
 * the scope can be narrowed while the store focuses on one production method.
 */
@Entity
@Table(name = "app_settings")
public class AppSetting extends PanacheEntityBase {

    @Id
    @Column(name = "key", nullable = false)
    public String key;

    @Column(name = "value", nullable = false)
    public String value;

    @Column(name = "updated_at")
    public OffsetDateTime updatedAt;

    @Column(name = "updated_by")
    public String updatedBy;

    public AppSetting() {}
}
