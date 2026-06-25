import { useState, useEffect } from "react";
import { API_BASE } from "../App";

const FALLBACK = {
  lexical: [
    { name: "url_length", description: "Total character length of the full URL", type: "numeric" },
    { name: "domain_length", description: "Character length of the registrable domain", type: "numeric" },
    { name: "shannon_entropy", description: "Shannon entropy of the second-level domain — high values indicate DGA usage", type: "numeric" },
    { name: "digit_ratio", description: "Proportion of digit characters in the domain", type: "ratio" },
    { name: "hyphen_count", description: "Number of hyphens in the domain", type: "count" },
    { name: "dot_count", description: "Number of dot characters in the full URL", type: "count" },
    { name: "subdomain_count", description: "Number of subdomain labels present", type: "count" },
    { name: "special_char_ratio", description: "Ratio of special punctuation characters in the URL", type: "ratio" },
    { name: "has_ip_address", description: "Binary: 1 if URL contains an IP address instead of domain name", type: "binary" },
    { name: "has_at_symbol", description: "Binary: 1 if URL contains @ symbol (credential redirect)", type: "binary" },
    { name: "has_double_slash", description: "Binary: 1 if double slash occurs after the protocol", type: "binary" },
    { name: "path_length", description: "Character length of the URL path component", type: "numeric" },
    { name: "suspicious_keyword_count", description: "Count of known phishing keywords found in the URL", type: "count" },
    { name: "tld_in_legitimate_list", description: "Binary: 1 if TLD is in the set of common legitimate TLDs", type: "binary" },
  ],
  structural: [
    { name: "domain_age_days", description: "Domain registration age in days — very low values indicate newly registered phishing domains", type: "numeric" },
    { name: "domain_expiry_days", description: "Days until domain registration expires", type: "numeric" },
    { name: "whois_available", description: "Binary: 1 if WHOIS record is publicly accessible", type: "binary" },
    { name: "dns_ttl_value", description: "DNS Time-To-Live in seconds — very low values indicate fast-flux evasion", type: "numeric" },
    { name: "has_mx_record", description: "Binary: 1 if a valid MX record exists", type: "binary" },
    { name: "has_spf_record", description: "Binary: 1 if a Sender Policy Framework TXT record is present", type: "binary" },
    { name: "dns_resolves", description: "Binary: 1 if the domain resolves to an IP address", type: "binary" },
    { name: "ns_count", description: "Count of nameserver records", type: "count" },
    { name: "ssl_valid", description: "Binary: 1 if a valid SSL/TLS certificate is present", type: "binary" },
    { name: "ssl_days_remaining", description: "Days until SSL certificate expires", type: "numeric" },
    { name: "ip_in_blacklist_asn", description: "Binary: 1 if IP resolves to a known high-risk ASN", type: "binary" },
    { name: "registrar_entropy", description: "Shannon entropy of the registrar name — high values suggest automated bulk registration", type: "numeric" },
    { name: "country_code_risk", description: "Binary: 1 if hosting country is in a high-risk jurisdiction", type: "binary" },
    { name: "nameserver_diversity", description: "Binary: 1 if nameservers are hosted across diverse providers", type: "binary" },
  ],
};

const importanceRanking = {
  shannon_entropy: { rank: 1, pipeline: "Lexical", weight: "~28%", note: "Dominant lexical feature — high entropy reveals DGA usage" },
  url_length: { rank: 2, pipeline: "Lexical", weight: "~14%", note: "Phishing URLs tend to be significantly longer" },
  digit_ratio: { rank: 3, pipeline: "Lexical", weight: "~12%", note: "Excessive digits correlate with obfuscation attempts" },
  domain_age_days: { rank: 1, pipeline: "Structural", weight: "~34%", note: "Most critical structural feature — phishing domains registered <48 hours before use" },
  ssl_valid: { rank: 2, pipeline: "Structural", weight: "~18%", note: "Absence of SSL is a strong phishing indicator" },
  dns_ttl_value: { rank: 3, pipeline: "Structural", weight: "~15%", note: "Low TTL exposes fast-flux anti-blacklisting technique" },
};

const typeColors = { numeric: "#6366f1", ratio: "#8b5cf6", count: "#06b6d4", binary: "#ec4899" };

export default function FeatureExplorer() {
  const [defs, setDefs] = useState(FALLBACK);
  const [activeTab, setActiveTab] = useState("lexical");
  const [search, setSearch] = useState("");

  useEffect(() => {
    fetch(`${API_BASE}/api/features/explain`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  const filtered = defs[activeTab]?.filter(f =>
    f.name.includes(search.toLowerCase()) || f.description.toLowerCase().includes(search.toLowerCase())
  ) ?? [];

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.5rem" }}>
        <div>
          <h2 style={{ margin: 0, color: "#e2e8f0", fontSize: "1.1rem" }}>Feature Definitions</h2>
          <p style={{ margin: 0, fontSize: "0.8rem", color: "#64748b" }}>14 lexical + 14 structural = 28 features (→ 25 post-correlation reduction)</p>
        </div>
        <input value={search} onChange={e => setSearch(e.target.value)}
          placeholder="Search features..."
          style={{ padding: "0.5rem 0.9rem", background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 6, color: "#e2e8f0", fontSize: "0.85rem", width: 200 }} />
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1.25rem" }}>
          {["lexical", "structural", "importance"].map(t => (
          <button key={t} onClick={() => setActiveTab(t)}
            style={{ padding: "0.5rem 1.1rem", background: activeTab === t ? "#6366f1" : "#1a1d2e", border: "1px solid #2d3148", borderRadius: 6, color: activeTab === t ? "#fff" : "#64748b", cursor: "pointer", fontSize: "0.85rem", fontWeight: activeTab === t ? 600 : 400 }}>
            {t === "lexical" ? "Lexical (14)" : t === "structural" ? "Structural (14)" : "Top Importance"}
          </button>
        ))}
      </div>

      {activeTab !== "importance" && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "0.75rem" }}>
          {filtered.map((feat, i) => {
            const imp = importanceRanking[feat.name];
            return (
              <div key={feat.name} style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 8, padding: "0.9rem 1rem" }}>
                <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: "0.4rem" }}>
                  <div>
                    <code style={{ fontSize: "0.82rem", color: "#a5b4fc", background: "#1e1b4b", padding: "2px 6px", borderRadius: 4 }}>{feat.name}</code>
                    {imp && <span style={{ marginLeft: "0.4rem", fontSize: "0.68rem", color: "#fbbf24", fontWeight: 700 }}>★ Rank #{imp.rank}</span>}
                  </div>
                  <span style={{ fontSize: "0.68rem", padding: "2px 6px", borderRadius: 10, background: `${typeColors[feat.type]}22`, color: typeColors[feat.type], whiteSpace: "nowrap" }}>
                    {feat.type}
                  </span>
                </div>
                <p style={{ margin: 0, fontSize: "0.8rem", color: "#94a3b8", lineHeight: 1.5 }}>{feat.description}</p>
                {imp && (
                  <p style={{ margin: "0.4rem 0 0", fontSize: "0.75rem", color: "#fbbf24", lineHeight: 1.4 }}>
                    ↑ Weight: {imp.weight} — {imp.note}
                  </p>
                )}
              </div>
            );
          })}
        </div>
      )}

      {activeTab === "importance" && (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
            {["Lexical", "Structural"].map(pipeline => {
              const entries = Object.entries(importanceRanking).filter(([, v]) => v.pipeline === pipeline).sort((a, b) => a[1].rank - b[1].rank);
              return (
                <div key={pipeline} style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem" }}>
                  <h3 style={{ margin: "0 0 1rem", fontSize: "0.9rem", color: "#94a3b8" }}>{pipeline} Pipeline — Feature Importance</h3>
                  {entries.map(([fname, meta]) => (
                    <div key={fname} style={{ marginBottom: "1rem" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <span style={{ fontSize: "0.82rem", fontWeight: 600, color: "#a5b4fc" }}>{fname}</span>
                        <span style={{ fontSize: "0.82rem", color: "#fbbf24", fontWeight: 700 }}>{meta.weight}</span>
                      </div>
                      <div style={{ background: "#0f1117", borderRadius: 4, height: 10, overflow: "hidden" }}>
                        <div style={{ width: meta.weight, height: "100%", background: pipeline === "Lexical" ? "#6366f1" : "#06b6d4", borderRadius: 4 }} />
                      </div>
                      <p style={{ margin: "4px 0 0", fontSize: "0.75rem", color: "#64748b" }}>{meta.note}</p>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>

          <div style={{ marginTop: "1.5rem", background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem" }}>
            <h3 style={{ margin: "0 0 0.75rem", fontSize: "0.9rem", color: "#94a3b8" }}>Key Finding: Feature-Classifier Interaction</h3>
            <p style={{ fontSize: "0.85rem", color: "#94a3b8", lineHeight: 1.7, margin: 0 }}>
              McNemar's test confirms <strong style={{ color: "#a5b4fc" }}>no statistically significant difference</strong> between Random Forest and XGBoost on the isolated lexical pipeline (p = 0.084).
              This is because RF's entropy-based splitting is inherently optimised for the high-dimensional, categorical nature of string-derived features.
              <br /><br />
              XGBoost only realises its mathematical advantage — specifically its L1/L2 regularisation and second-order gradient optimisation — when introduced to the <strong style={{ color: "#06b6d4" }}>sparse, continuous, non-linear structural metadata</strong> in Pipelines B and C (p &lt; 0.001).
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
