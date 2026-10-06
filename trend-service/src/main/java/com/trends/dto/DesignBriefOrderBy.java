package com.trends.dto;

public enum DesignBriefOrderBy {
    CREATED_AT_DESC("db.createdAt DESC, db.id DESC"),
    CREATED_AT_ASC("db.createdAt ASC, db.id ASC"),
    CONFIDENCE_DESC("db.confidence DESC, db.createdAt DESC, db.id DESC"),
    CONFIDENCE_ASC("db.confidence ASC, db.createdAt ASC, db.id ASC");

    private final String jpqlOrderBy;

    DesignBriefOrderBy(String jpqlOrderBy) {
        this.jpqlOrderBy = jpqlOrderBy;
    }

    public String toJpql() {
        return jpqlOrderBy;
    }

    public String toPanacheOrderBy() {
        return jpqlOrderBy.replace("db.", "");
    }
}
