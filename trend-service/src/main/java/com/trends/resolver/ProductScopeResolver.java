package com.trends.resolver;

import com.trends.config.ProductScope;
import io.quarkus.security.identity.SecurityIdentity;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.graphql.GraphQLApi;
import org.eclipse.microprofile.graphql.GraphQLException;
import org.eclipse.microprofile.graphql.Mutation;
import org.eclipse.microprofile.graphql.Query;
import org.jboss.logging.Logger;

import java.util.List;

/**
 * The product types the pipeline produces right now.
 *
 * <p>The store focuses on one production method at a time, and the scope can
 * be changed here without a redeploy.
 */
@GraphQLApi
@ApplicationScoped
public class ProductScopeResolver {

    private static final Logger LOG = Logger.getLogger(ProductScopeResolver.class);

    @Inject
    ProductScope productScope;

    @Inject
    SecurityIdentity securityIdentity;

    @Query("enabledProductTypes")
    public List<String> getEnabledProductTypes() throws GraphQLException {
        requireAuthentication();
        LOG.debugf("Fetching the enabled product types");
        return productScope.enabledProductTypes();
    }

    @Mutation("setEnabledProductTypes")
    public List<String> setEnabledProductTypes(List<String> types) throws GraphQLException {
        requireAuthentication();

        if (types == null || types.isEmpty()) {
            throw new GraphQLException("At least one product type is required");
        }

        for (String type : types) {
            if (!ProductScope.KNOWN_PRODUCT_TYPES.contains(type)) {
                throw new GraphQLException("Unknown product type: " + type);
            }
        }

        List<String> before = productScope.enabledProductTypes();
        productScope.setEnabledProductTypes(types, caller());
        LOG.infof("Changed the product type scope from %s to %s", before, types);
        return productScope.enabledProductTypes();
    }

    private String caller() {
        if (securityIdentity == null) {
            return null;
        }
        return securityIdentity.getPrincipal() != null
                ? securityIdentity.getPrincipal().getName()
                : null;
    }

    private void requireAuthentication() throws GraphQLException {
        if (securityIdentity == null || securityIdentity.isAnonymous()) {
            throw new GraphQLException("Authentication required");
        }
    }
}
