package com.trends.domain;

import io.quarkus.hibernate.orm.panache.PanacheEntityBase;
import jakarta.persistence.*;
import java.time.OffsetDateTime;
import java.util.List;

@Entity
@Table(name = "design_briefs")
public class DesignBrief extends PanacheEntityBase {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    public Integer id;

    @Column(name = "concept", nullable = false, columnDefinition = "TEXT")
    public String concept;

    @Column(name = "product_type", nullable = false, length = 50)
    public String productType;

    @Column(name = "specific_products", columnDefinition = "TEXT[]")
    public String[] specificProducts;

    @Column(name = "audience", columnDefinition = "TEXT")
    public String audience;

    @Column(name = "visual_style", columnDefinition = "TEXT")
    public String visualStyle;

    @Column(name = "confidence", nullable = false)
    public Integer confidence;

    @Column(name = "reasoning", columnDefinition = "TEXT")
    public String reasoning;

    @Column(name = "llm_model", length = 100)
    public String llmModel;

    @Column(name = "batch_id", length = 50)
    public String batchId;

    @Column(name = "created_at")
    public OffsetDateTime createdAt;

    @Column(name = "image_key_raw")
    public String imageKeyRaw;

    @Column(name = "image_key_transparent")
    public String imageKeyTransparent;

    @Column(name = "regeneration_feedback", columnDefinition = "TEXT")
    public String regenerationFeedback;

    @Column(name = "layout_type", length = 30)
    public String layoutType;

    @Column(name = "headline_text", columnDefinition = "TEXT")
    public String headlineText;

    @Column(name = "tagline_text", columnDefinition = "TEXT")
    public String taglineText;

    @Column(name = "font_color", length = 20)
    public String fontColor;

    @Column(name = "scene_description", columnDefinition = "TEXT")
    public String sceneDescription;

    @Column(name = "personal_use", nullable = false)
    public Boolean personalUse = false;

    @OneToMany(mappedBy = "designBrief", fetch = FetchType.LAZY)
    public List<DesignBriefSource> sources;

    public DesignBrief() {}
}
