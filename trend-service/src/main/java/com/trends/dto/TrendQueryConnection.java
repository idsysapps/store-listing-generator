package com.trends.dto;

import java.util.List;

public class TrendQueryConnection {
    private List<TrendQueryEdge> edges;
    private PageInfo pageInfo;

    public TrendQueryConnection() {}

    public TrendQueryConnection(List<TrendQueryEdge> edges, PageInfo pageInfo) {
        this.edges = edges;
        this.pageInfo = pageInfo;
    }

    public List<TrendQueryEdge> getEdges() {
        return edges;
    }

    public void setEdges(List<TrendQueryEdge> edges) {
        this.edges = edges;
    }

    public PageInfo getPageInfo() {
        return pageInfo;
    }

    public void setPageInfo(PageInfo pageInfo) {
        this.pageInfo = pageInfo;
    }
}
