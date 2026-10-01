package com.trends.dto;

public class SystemStatus {

    private int candidatesPending;
    private int candidatesPromoted;
    private int candidatesArchived;
    private int activeSeedsCount;
    private int designBriefsTotal;
    private int designBriefsWithImages;
    private int designBriefsPendingImages;
    private int designBriefsPendingRegen;

    public SystemStatus() {}

    public SystemStatus(int candidatesPending, int candidatesPromoted, int candidatesArchived,
            int activeSeedsCount, int designBriefsTotal, int designBriefsWithImages,
            int designBriefsPendingImages, int designBriefsPendingRegen) {
        this.candidatesPending = candidatesPending;
        this.candidatesPromoted = candidatesPromoted;
        this.candidatesArchived = candidatesArchived;
        this.activeSeedsCount = activeSeedsCount;
        this.designBriefsTotal = designBriefsTotal;
        this.designBriefsWithImages = designBriefsWithImages;
        this.designBriefsPendingImages = designBriefsPendingImages;
        this.designBriefsPendingRegen = designBriefsPendingRegen;
    }

    public int getCandidatesPending() { return candidatesPending; }
    public int getCandidatesPromoted() { return candidatesPromoted; }
    public int getCandidatesArchived() { return candidatesArchived; }
    public int getActiveSeedsCount() { return activeSeedsCount; }
    public int getDesignBriefsTotal() { return designBriefsTotal; }
    public int getDesignBriefsWithImages() { return designBriefsWithImages; }
    public int getDesignBriefsPendingImages() { return designBriefsPendingImages; }
    public int getDesignBriefsPendingRegen() { return designBriefsPendingRegen; }
}
