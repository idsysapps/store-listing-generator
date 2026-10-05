package com.trends.dto;

import java.util.List;

public class DesignBriefConnection {
    private List<DesignBriefSummary> items;
    private PageInfo pageInfo;

    public DesignBriefConnection() {}

    public DesignBriefConnection(List<DesignBriefSummary> items, PageInfo pageInfo) {
        this.items = items;
        this.pageInfo = pageInfo;
    }

    public List<DesignBriefSummary> getItems() {
        return items;
    }

    public void setItems(List<DesignBriefSummary> items) {
        this.items = items;
    }

    public PageInfo getPageInfo() {
        return pageInfo;
    }

    public void setPageInfo(PageInfo pageInfo) {
        this.pageInfo = pageInfo;
    }
}
