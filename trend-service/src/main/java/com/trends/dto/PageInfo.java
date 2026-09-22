package com.trends.dto;

public class PageInfo {
    private Boolean hasNextPage;
    private Boolean hasPreviousPage;
    private Integer totalCount;

    public PageInfo() {}

    public PageInfo(Boolean hasNextPage, Boolean hasPreviousPage, Integer totalCount) {
        this.hasNextPage = hasNextPage;
        this.hasPreviousPage = hasPreviousPage;
        this.totalCount = totalCount;
    }

    public Boolean getHasNextPage() {
        return hasNextPage;
    }

    public void setHasNextPage(Boolean hasNextPage) {
        this.hasNextPage = hasNextPage;
    }

    public Boolean getHasPreviousPage() {
        return hasPreviousPage;
    }

    public void setHasPreviousPage(Boolean hasPreviousPage) {
        this.hasPreviousPage = hasPreviousPage;
    }

    public Integer getTotalCount() {
        return totalCount;
    }

    public void setTotalCount(Integer totalCount) {
        this.totalCount = totalCount;
    }
}
