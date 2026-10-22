package com.trends.config;

import com.trends.domain.AppSetting;
import com.trends.repository.AppSettingRepository;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Which product types the pipeline produces right now.
 *
 * <p>The store focuses on one production method at a time: the DTF apparel
 * batches are the ones that sell, and the sublimation ones wait. The scope is
 * resolved in one place, so the API, the prompts and the queries all agree on
 * what is being produced.
 *
 * <p>The stored setting wins over the environment, so the scope can be changed
 * without a redeploy. A setting that names nothing usable falls back to the
 * environment, and an environment that names nothing falls back to every
 * known type.
 */
@ApplicationScoped
public class ProductScope {

    private static final Logger LOG = Logger.getLogger(ProductScope.class);

    public static final List<String> KNOWN_PRODUCT_TYPES =
            List.of("dtf_apparel", "sublimation", "sticker_vinyl");

    public static final String SETTING_KEY = "enabled_product_types";

    /**
     * The scope from the environment.
     *
     * <p>Optional, because an empty value is not a value: SmallRye refuses
     * to start when a property is defined as the empty string, and the chart
     * ships the key empty when a cluster does not set it (issue #181).
     */
    @ConfigProperty(name = "ENABLED_PRODUCT_TYPES")
    Optional<String> envValue;

    @Inject
    AppSettingRepository appSettingRepository;

    /** The product types that are produced right now. */
    public List<String> enabledProductTypes() {
        AppSetting stored = null;
        try {
            stored = appSettingRepository.findByKey(SETTING_KEY);
        } catch (Exception e) {
            // A database that cannot be reached must not stop the caller:
            // the environment is the fallback.
            LOG.warnf("Could not read the stored product type scope: %s", e.getMessage());
        }

        if (stored != null) {
            List<String> types = parse(stored.value);
            if (!types.isEmpty()) {
                return types;
            }
            LOG.warnf("The stored product type scope names no known type, falling back to the environment");
        }

        List<String> types = parse(envValue.orElse(""));
        if (!types.isEmpty()) {
            return types;
        }

        return KNOWN_PRODUCT_TYPES;
    }

    /** Store the product types that are produced right now. */
    public void setEnabledProductTypes(List<String> types, String updatedBy) {
        appSettingRepository.set(SETTING_KEY, String.join(",", types), updatedBy);
    }

    /** The product types named in a setting or an environment variable. */
    public List<String> parse(String raw) {
        List<String> types = new ArrayList<>();
        if (raw == null || raw.isBlank()) {
            return types;
        }
        for (String part : raw.split(",")) {
            String value = part.trim().toLowerCase();
            if (value.isEmpty()) {
                continue;
            }
            if (!KNOWN_PRODUCT_TYPES.contains(value)) {
                LOG.warnf("Unknown product type %s, ignored", value);
                continue;
            }
            if (!types.contains(value)) {
                types.add(value);
            }
        }
        return types;
    }
}
