package com.trends.dto;

import com.trends.domain.TrendQuery;

public class TrendQueryEdge {
    private TrendQuery node;
    private String cursor;

    public TrendQueryEdge() {}

    public TrendQueryEdge(TrendQuery node, String cursor) {
        this.node = node;
        this.cursor = cursor;
    }

    public TrendQuery getNode() {
        return node;
    }

    public void setNode(TrendQuery node) {
        this.node = node;
    }

    public String getCursor() {
        return cursor;
    }

    public void setCursor(String cursor) {
        this.cursor = cursor;
    }
}
