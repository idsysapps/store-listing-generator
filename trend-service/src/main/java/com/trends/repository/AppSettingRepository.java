package com.trends.repository;

import com.trends.domain.AppSetting;
import io.quarkus.hibernate.orm.panache.PanacheRepository;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.transaction.Transactional;

import java.time.OffsetDateTime;

@ApplicationScoped
public class AppSettingRepository implements PanacheRepository<AppSetting> {

    public AppSetting findByKey(String key) {
        return find("key", key).firstResult();
    }

    @Transactional
    public AppSetting set(String key, String value, String updatedBy) {
        AppSetting setting = findByKey(key);
        if (setting == null) {
            setting = new AppSetting();
            setting.key = key;
            setting.value = value;
            setting.updatedBy = updatedBy;
            setting.updatedAt = OffsetDateTime.now();
            persist(setting);
            return setting;
        }
        setting.value = value;
        setting.updatedBy = updatedBy;
        setting.updatedAt = OffsetDateTime.now();
        return setting;
    }
}
