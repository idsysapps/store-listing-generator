package com.trends.dto;

public class ImageGenerationStatus {
    private int briefId;
    private String phase;
    private int step;
    private int totalSteps;
    private double elapsedSeconds;
    private double estimatedTotalSeconds;
    private double remainingSeconds;

    public ImageGenerationStatus() {}

    public int getBriefId() {
        return briefId;
    }

    public void setBriefId(int briefId) {
        this.briefId = briefId;
    }

    public String getPhase() {
        return phase;
    }

    public void setPhase(String phase) {
        this.phase = phase;
    }

    public int getStep() {
        return step;
    }

    public void setStep(int step) {
        this.step = step;
    }

    public int getTotalSteps() {
        return totalSteps;
    }

    public void setTotalSteps(int totalSteps) {
        this.totalSteps = totalSteps;
    }

    public double getElapsedSeconds() {
        return elapsedSeconds;
    }

    public void setElapsedSeconds(double elapsedSeconds) {
        this.elapsedSeconds = elapsedSeconds;
    }

    public double getEstimatedTotalSeconds() {
        return estimatedTotalSeconds;
    }

    public void setEstimatedTotalSeconds(double estimatedTotalSeconds) {
        this.estimatedTotalSeconds = estimatedTotalSeconds;
    }

    public double getRemainingSeconds() {
        return remainingSeconds;
    }

    public void setRemainingSeconds(double remainingSeconds) {
        this.remainingSeconds = remainingSeconds;
    }
}
