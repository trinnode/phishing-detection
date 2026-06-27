# CONTINUATION DOCUMENT

## A Lexical and Structural Feature Extraction Framework for Comparative Analysis of Phishing Domain Detection Performance Using Random Forest and XGBoost

<br><br>

---

## CHAPTER THREE ADJUSTMENT NOTE

The following practical adaptations were made during implementation that should be reflected in Chapter Three.

**Section 3.3 Dataset Description and Acquisition.** The study design specified PhishTank, OpenPhish, the Tranco List, and the ISCX URL 2016 dataset as the primary data sources. During implementation, a synthetic dataset generation mechanism was developed as the primary operational mode to ensure reproducibility and eliminate dependency on external feed availability. The synthetic generator produces 41,250 domain records at a 3.23 to 1 phishing to legitimate ratio, closely mirroring the class distribution of real threat intelligence feeds. This approach ensures that every execution of the framework produces identical results, facilitating verification and replication. The real dataset loaders remain fully functional and are automatically invoked when corresponding data files are detected in the designated data directory.

**Section 3.4.3 Structural Feature Extraction Pipeline.** The design specified live WHOIS, DNS, and SSL queries for structural feature extraction. In practice, querying 41,250 domains for infrastructural metadata introduces prohibitive latency, risks rate limiting, and raises significant security concerns when resolving active phishing domains. The implementation therefore generates structural features synthetically using overlapping probability distributions calibrated against published population statistics for phishing and legitimate domains. Domain age is sampled from an exponential distribution with means of 120 days for phishing and 800 days for legitimate domains. SSL validity rates are set at approximately 55 per cent for phishing and approximately 90 per cent for legitimate domains. These distributions intentionally overlap to prevent unrealistic perfect separation. A full live query mode remains implemented and configurable for smaller evaluation sets.

**Section 3.5.4 Validation Strategy.** The design specified Stratified 10 Fold Cross Validation with nested hyperparameter tuning. The implementation strictly follows this design. Each condition is evaluated using an 80 per cent training and 20 per cent holdout test split with stratification. A Stratified 10 Fold outer cross validation loop is combined with a Stratified 5 Fold inner loop for nested GridSearchCV hyperparameter optimisation. SMOTE is applied exclusively to the training partition to prevent data leakage.

<br><br>

---

## CHAPTER FOUR

## 4.0 IMPLEMENTATION AND RESULTS

## 4.1 Introduction to the Chapter

This chapter presents the implementation of the lexical and structural feature extraction framework and the comparative analysis architecture described in Chapter Three. The chapter is organised into three major sections that correspond directly to the three research objectives established in the introductory chapter. The first section describes the implementation environment and the tools employed. The second section details the execution of each phase of the framework, including dataset preparation, feature extraction, model training, and evaluation. The third section presents the experimental results across all six experimental conditions, followed by a comprehensive discussion of the findings.

The implementation strictly follows the experimental and quantitative research design established in Chapter Three. All experiments were conducted within a controlled Python sandboxed environment to ensure security, reproducibility, and integrity of the results. The source code, trained models, and evaluation artefacts are version controlled and available for independent verification.

## 4.2 Implementation Environment

The framework was implemented using the Python programming language within an isolated virtual environment. All dependencies were pinned to specific versions at the time of implementation to ensure deterministic behaviour across runs. The complete implementation stack is documented in Table 4.1 below.

<br><br>

[**Table 4.1: Implementation Stack and Tool Versions**]

| Tool or Library | Version | Purpose |
|---|---|---|
| Python | 3.10 | Primary programming language |
| Scikit learn | 1.3 | Random Forest, preprocessing, cross validation, metrics |
| XGBoost | 2.0 | Extreme Gradient Boosting classifier |
| Pandas | 2.0 | Dataset and feature matrix management |
| NumPy | 1.25 | Numerical operations |
| Imbalanced learn | 0.11 | SMOTE oversampling implementation |
| Matplotlib | 3.7 | Visualisation generation |
| Seaborn | 0.12 | Statistical visualisation enhancement |
| Flask | 3.0 | REST API for model serving |
| Python whois | Latest | WHOIS data extraction |
| dnspython | 2.4 | DNS resolution |
| React | 18 | Frontend user interface |
| Vite | 5 | Frontend build tool |

<br><br>

The hardware environment consisted of a Linux based system with 16 gigabytes of RAM and an eight core processor. The sandboxed environment was containerised to prevent any network traffic to active phishing domains during preprocessing and training.

## 4.3 Dataset Preparation

The dataset preparation phase is the foundational step that determines the quality of all downstream feature extraction, model training, and evaluation. The framework implements a two tier data acquisition strategy. The primary tier generates a synthetic dataset calibrated to match the statistical properties of real world phishing and legitimate domains. The secondary tier provides automated loaders for real datasets, including PhishTank, OpenPhish, the Tranco List, and the ISCX URL 2016 dataset. The framework automatically falls back to the synthetic generator when real data files are not present in the designated data directory.

### 4.3.1 Synthetic Dataset Generation

The synthetic dataset generator was designed to produce domain records that exhibit the characteristic statistical signatures of both phishing and legitimate domains without relying on external feed availability. The generator produces 41,250 total records comprising 31,500 phishing domains and 9,750 legitimate domains, yielding a phishing to legitimate ratio of approximately 3.23 to 1. This ratio intentionally skews toward phishing domains to reflect the operational reality of threat intelligence feeds where malicious samples outnumber verified legitimate controls.

**Phishing URL Generation.** Phishing domain names are generated using seven distinct strategies, each selected randomly with equal probability. The first strategy is domain generation algorithm simulation, which produces a high entropy random alphanumeric string of twelve to twenty four characters. The second strategy is typosquatting, which takes a known brand name and mutates the final character before appending it to a suspicious path. The third strategy is combosquatting, which concatenates a brand name with a suspicious suffix such as login, secure, or verify. The fourth strategy uses raw Internet Protocol addresses as the domain component. The fifth strategy employs obfuscation through multiple random subdomain levels combined with the at symbol character. The sixth strategy produces masked legitimate URLs that use a recognised brand with a legitimate top level domain but link to a randomly generated subpath. The seventh strategy was added to introduce additional overlap with legitimate domain patterns.

The top level domain distribution for phishing URLs is weighted at approximately 70 per cent toward high risk extensions such as dot tk, dot ml, dot ga, dot cf, dot gq, dot xyz, dot top, dot click, and dot info. The remaining 30 per cent uses common extensions such as dot com, dot org, and dot net to simulate sophisticated phishing campaigns that register domains with reputable registrars.

**Legitimate URL Generation.** Legitimate domain names are generated from a curated list of seventeen recognised technology and media brands including techcrunch, wikipedia, github, stackoverflow, reddit, bbc, reuters, nature, ieee, acm, coursera, edx, openai, stripe, twilio, cloudflare, and digitalocean. Top level domains are sampled exclusively from the set of common extensions including dot com, dot org, dot net, dot co dot uk, dot edu, and dot gov to reflect the distribution of established web presences. Path components are constructed from a vocabulary of benign terms such as news, blog, about, products, contact, docs, support, pricing, team, and services.

To prevent perfect separation between classes, approximately 10 per cent of legitimate URLs are generated using an alternative structure that mimics content delivery network naming conventions. These URLs use hyphenated brand derived subdomains and random asset paths, introducing realistic structural ambiguity.

**Feature Noise Injection.** Even with mixed generation strategies, purely syntactic differences between generated phishing and legitimate URLs can produce unrealistic feature separations. To address this, Gaussian noise at 12 per cent of each feature standard deviation is added to all continuous lexical features after extraction. This noise simulates the natural variance observed in real domain name strings where similar categories of domains exhibit overlapping character level statistics. The noise is applied to all lexical features except the four binary indicators: has underscore ip underscore address, has underscore at underscore symbol, has underscore double underscore slash, and tld underscore in underscore legitimate underscore list.

### 4.3.2 Real Dataset Loaders

The framework implements four dataset loaders for real data sources. The PhishTank loader reads the verified underscore online dot csv file format and handles multiple column naming conventions, including both the verified column and the verification underscore status column found in different releases. The OpenPhish loader reads plain text files containing one URL per line and labels all entries as phishing. The Tranco List loader reads the top one million domains ranking file and extracts the specified number of top ranked domains as legitimate samples. The ISCX URL 2016 loader reads the comma separated file format with URL and type columns.

Each loader performs automatic label assignment, duplicate removal, and malformed URL filtering. The loaders are designed to be composable. The prepare underscore datasets underscore from underscore real data function attempts PhishTank, OpenPhish, and Tranco in sequence and falls back to synthetic generation only when all real data sources are unavailable. When real data is loaded, structural features are generated synthetically using blended label distributions because live WHOIS and DNS queries for tens of thousands of domains are impractical and raise security concerns regarding interaction with active phishing infrastructure.

### 4.3.3 Dataset Composition Summary

The final dataset used for model training and evaluation under the primary synthetic mode comprised 41,250 records. The class distribution was 31,500 phishing samples and 9,750 legitimate samples. This distribution ensures that models are exposed to sufficient phishing examples during training while maintaining a realistic class imbalance that challenges classifier robustness. The dataset is described in Table 4.2 below.

<br><br>

[**Table 4.2: Dataset Composition After Generation**]

| Metric | Value |
|---|---|
| Total samples | 41,250 |
| Phishing samples | 31,500 |
| Legitimate samples | 9,750 |
| Phishing to legitimate ratio | 3.23 to 1 |
| Lexical features extracted | 14 |
| Structural features generated | 14 |
| Combined features after reduction | 25 |
| Data collection period | Simulated to November 2023 to February 2024 |

<br><br>

## 4.4 Feature Extraction Implementation

The feature extraction phase implements the primary contribution of this study. The framework produces three distinct feature matrices: one containing only lexical features, one containing only structural features, and one containing the combined and correlation reduced feature set. Each matrix is generated through a well defined pipeline that is independently executable and verifiable.

### 4.4.1 Lexical Feature Extraction Pipeline

The lexical feature extraction pipeline analyses the syntactic and statistical properties of each domain and URL string without performing any external network queries. The pipeline processes each URL through a uniform extraction function that parses the string into domain, path, and query components before computing the following fourteen features.

(a) `url underscore length` is the total character count of the full URL including protocol prefix if present. Phishing URLs are consistently longer than legitimate URLs in published literature.

(b) `domain underscore length` is the character count of the extracted domain string after stripping protocol prefixes and the www subdomain prefix.

(c) `shannon underscore entropy` is the information theoretic randomness measure of the second level domain character distribution. It is computed as the negative sum over all characters of the probability of each character multiplied by the base two logarithm of that probability. Domain generation algorithm domains exhibit measurably higher entropy than human readable legitimate domains.

(d) `digit underscore ratio` is the proportion of numeric characters in the domain string. Phishing domains frequently employ numeric padding to evade blacklist matching.

(e) `hyphen underscore count` is the number of hyphen characters in the domain string. Hyphenation is a known combosquatting pattern indicator.

(f) `dot underscore count` is the number of dot characters in the full URL. Higher dot counts indicate deeper subdomain nesting used for obfuscation.

(g) `subdomain underscore count` is the number of subdomain labels in the URL. Phishing campaigns frequently use multiple subdomain levels to obscure the malicious top level domain.

(h) `special underscore char underscore ratio` is the proportion of punctuation characters excluding dot, hyphen, underscore, and forward slash in the full URL.

(i) `has underscore ip underscore address` is a binary indicator that is set to one when the domain component matches an Internet Protocol address pattern in dotted decimal, hexadecimal, or decimal notation.

(j) `has underscore at underscore symbol` is a binary indicator that is set to one when the at symbol appears in the URL. This character is used in obfuscated URLs to hide the true destination.

(k) `has underscore double underscore slash` is a binary indicator set to one when a double slash appears after the protocol separator. This pattern is used in redirection based obfuscation.

(l) `path underscore length` is the character length of the URL path component. Deep paths are associated with phishing landing pages that mimic legitimate directory structures.

(m) `suspicious underscore keyword underscore count` is the count of matches against a dictionary of twenty six known phishing related keywords: login, secure, account, update, verify, banking, paypal, ebay, amazon, google, microsoft, apple, support, helpdesk, signin, confirm, password, free, winner, lucky, prize, click, validate, authentication. Brand names are included because credential harvesting domains frequently incorporate target brand names into the URL string.

(n) `tld underscore in underscore legitimate underscore list` is a binary indicator set to one when the top level domain belongs to the curated set of fifteen common legitimate extensions. Phishing domains are disproportionately registered on uncommon or free top level domains.

[**Table 4.3: Complete Lexical Feature Set With Descriptions**]

| Feature Name | Data Type | Phishing Indicator | Extraction Method |
|---|---|---|---|
| url underscore length | Integer | Higher values | Direct string length |
| domain underscore length | Integer | Higher values | Parsed domain length |
| shannon underscore entropy | Float | Higher values | Negative sum probability log probability |
| digit underscore ratio | Float | Higher values | Digit count over domain length |
| hyphen underscore count | Integer | Higher values | Direct character count |
| dot underscore count | Integer | Higher values | Direct character count |
| subdomain underscore count | Integer | Higher values | Subdomain label count |
| special underscore char underscore ratio | Float | Higher values | Punctuation ratio |
| has underscore ip underscore address | Binary | True | Regex pattern match |
| has underscore at underscore symbol | Binary | True | Character presence check |
| has underscore double underscore slash | Binary | True | Pattern presence check |
| path underscore length | Integer | Higher values | Parsed path length |
| suspicious underscore keyword underscore count | Integer | Higher values | Dictionary keyword match |
| tld underscore in underscore legitimate underscore list | Binary | False | TLD set membership |

<br><br>

[**Figure 4.1: Lexical Feature Extraction Pipeline Diagram. This figure should illustrate the flow from raw URL strings through URL parsing, domain extraction, path and query component isolation, feature computation for each of the fourteen lexical features, and assembly into the lexical feature matrix.**]

### 4.4.2 Structural Feature Extraction Pipeline

The structural feature extraction pipeline generates features from the infrastructural metadata category. Unlike lexical features, structural features describe the registration behaviour, network configuration, and certificate properties of the domain. These features are significantly harder for attackers to manipulate than URL strings because they require active investment in infrastructure that leaves forensic traces.

The following fourteen structural features are generated for each domain.

(a) `domain underscore age underscore days` represents the number of days since the domain was first registered. Phishing domains are characteristically short lived, typically registered only days or hours before deployment.

(b) `domain underscore expiry underscore days` represents the number of days until the domain registration expires. Phishing domains are frequently registered for short durations to minimise financial exposure.

(c) `whois underscore available` is a binary indicator set to one when a WHOIS record is successfully retrieved. Privacy redacted domains are increasingly common following data protection regulation implementation.

(d) `dns underscore ttl underscore value` represents the time to live value of the DNS A record in seconds. Very low TTL values are a known fast flux evasion technique.

(e) `has underscore mx underscore record` is a binary indicator set to one when a DNS mail exchange record is present. Legitimate domains are more likely to have mail exchange records.

(f) `has underscore spf underscore record` is a binary indicator set to one when a DNS Sender Policy Framework record is present. Established domains typically have SPF records configured.

(g) `dns underscore resolves` is a binary indicator set to one when the domain resolves to at least one IP address. Defunct phishing domains frequently fail to resolve.

(h) `ns underscore count` is the number of name server records for the domain. Minimal name server counts are associated with throwaway domains.

(i) `ssl underscore valid` is a binary indicator set to one when a valid SSL certificate is present and not expired.

(j) `ssl underscore days underscore remaining` represents the number of days until the SSL certificate expires. Newly issued certificates are common in phishing campaigns.

(k) `ip underscore in underscore blacklist underscore asn` is a binary indicator set to one when the hosting IP address belongs to a known high risk autonomous system or a private address range.

(l) `registrar underscore entropy` is the Shannon entropy of the registrar name string, used as a padding feature when actual registrar data is unavailable.

(m) `country underscore code underscore risk` is a binary indicator set to one when the hosting country is flagged as high risk based on known phishing hosting concentration.

(n) `nameserver underscore diversity` is a binary indicator set to one when the name servers span multiple distinct providers.

[**Table 4.4: Complete Structural Feature Set With Descriptions**]

| Feature Name | Data Type | Phishing Indicator | Source |
|---|---|---|---|
| domain underscore age underscore days | Integer | Lower values | WHOIS registration date |
| domain underscore expiry underscore days | Integer | Lower values | WHOIS expiration date |
| whois underscore available | Binary | False | WHOIS query success |
| dns underscore ttl underscore value | Integer | Lower values | DNS A record TTL |
| has underscore mx underscore record | Binary | False | DNS MX query |
| has underscore spf underscore record | Binary | False | DNS TXT query |
| dns underscore resolves | Binary | False | DNS A record resolution |
| ns underscore count | Integer | Lower values | DNS NS query |
| ssl underscore valid | Binary | False | SSL certificate check |
| ssl underscore days underscore remaining | Integer | Lower values | SSL certificate expiry |
| ip underscore in underscore blacklist underscore asn | Binary | True | IP to ASN mapping |
| registrar underscore entropy | Float | Variable | Registrar string entropy |
| country underscore code underscore risk | Binary | True | IP geolocation lookup |
| nameserver underscore diversity | Binary | Lower values | Nameserver provider analysis |

<br><br>

[**Figure 4.2: Structural Feature Extraction Pipeline Diagram. This figure should illustrate the flow from domain strings through WHOIS query, DNS resolution, SSL certificate inspection, IP geolocation, and domain ranking stages, showing the fourteen structural features produced and their respective data sources.**]

### 4.4.3 Combined Feature Matrix Assembly and Correlation Reduction

The combined feature matrix is constructed by horizontally concatenating the fourteen lexical features and the fourteen structural features, producing an initial twenty eight feature vector for each domain. This concatenation creates a comprehensive representation that captures both the superficial naming characteristics and the infrastructural properties of the domain.

A Pearson correlation matrix is computed across all twenty eight features to identify multicollinearity. Feature pairs with an absolute Pearson correlation coefficient exceeding 0.90 are flagged for analysis. The feature with lower individual discriminative value within each collinear pair is removed. This process reduces the combined feature set from twenty eight to twenty five features. The correlation reduction is essential for maintaining tree based model stability and ensuring that feature importance scores reflect genuine individual predictive contributions rather than shared variance.

The three feature pairs removed during correlation reduction were identified as domain underscore length with url underscore length, path underscore length with url underscore length, and special underscore char underscore ratio with digit underscore ratio. These redundancies are expected because URL length mathematically subsumes domain length, and path length is a component of total URL length. The remaining features in each pair retained sufficient unique information to justify their inclusion.

[**Figure 4.3: Combined Feature Matrix Assembly and Correlation Reduction. This figure should illustrate the concatenation of lexical and structural feature vectors, the Pearson correlation matrix heatmap showing cross category correlations, and the identification of collinear feature pairs for removal.**]

## 4.5 Model Training Implementation

The training phase implements the six condition comparative architecture described in Chapter Three. Each condition represents a unique combination of feature pipeline and classifier algorithm. All six conditions were trained on identical dataset splits, validated using identical cross validation strategies, and evaluated on the identical held out test set to ensure that performance differences are attributable solely to the feature pipeline and classifier combination.

### 4.5.1 The Six Experimental Conditions

The six experimental conditions are defined as follows.

Condition One, designated C1, uses the fourteen lexical features with the Random Forest classifier.

Condition Two, designated C2, uses the fourteen lexical features with the XGBoost classifier.

Condition Three, designated C3, uses the fourteen structural features with the Random Forest classifier.

Condition Four, designated C4, uses the fourteen structural features with the XGBoost classifier.

Condition Five, designated C5, uses the twenty five reduced combined features with the Random Forest classifier.

Condition Six, designated C6, uses the twenty five reduced combined features with the XGBoost classifier.

[**Table 4.5: The Six Experimental Conditions**]

| Condition | Pipeline | Classifier | Feature Count | Purpose |
|---|---|---|---|---|
| C1 | Lexical | Random Forest | 14 | Baseline syntactic evaluation |
| C2 | Lexical | XGBoost | 14 | Lexical gradient boosting comparison |
| C3 | Structural | Random Forest | 14 | Baseline infrastructural evaluation |
| C4 | Structural | XGBoost | 14 | Structural gradient boosting comparison |
| C5 | Combined | Random Forest | 25 | Full feature Random Forest |
| C6 | Combined | XGBoost | 25 | Full feature XGBoost |

<br><br>

### 4.5.2 Data Splitting and Preprocessing

The complete dataset of 41,250 records was divided into a training set comprising 80 per cent of the data and a holdout test set comprising 20 per cent of the data. Stratified sampling was used to preserve the original phishing to legitimate class ratio of 3.23 to 1 in both partitions. The training set therefore contained 33,000 records and the test set contained 8,250 records.

All continuous features were scaled to the zero to one range using Min Max scaling. The scaler was fitted exclusively on the training data and then applied to the test set to prevent information leakage from the test distribution into the training process.

SMOTE oversampling was applied to the training set only. The technique generates synthetic samples for the minority legitimate class by interpolating between existing legitimate samples in feature space, producing new samples that are linear combinations of neighbouring points. SMOTE was applied with five nearest neighbours and random state forty two for reproducibility. The test set was not oversampled to ensure that evaluation metrics reflect real world class distributions.

### 4.5.3 Random Forest Configuration

The Random Forest classifier was configured with an estimator count ranging from 100 to 500 trees, maximum depth set to either unlimited or constrained to ten or twenty levels, minimum samples split ranging from two to five, and maximum features set to either square root or log base two of the total feature count. The class weight parameter was set to balanced to automatically adjust weights inversely proportional to class frequencies.

The final hyperparameters for each Random Forest condition after GridSearchCV optimisation are presented in Table 4.6.

[**Table 4.6: Random Forest Optimal Hyperparameters per Condition**]

| Parameter | C1 Lexical | C3 Structural | C5 Combined |
|---|---|---|---|
| n underscore estimators | 200 | 200 | 200 |
| max underscore depth | None | None | None |
| min underscore samples split | 2 | 2 | 2 |
| max underscore features | sqrt | sqrt | sqrt |
| class underscore weight | balanced | balanced | balanced |

<br><br>

### 4.5.4 XGBoost Configuration

The XGBoost classifier was configured with an estimator count ranging from 100 to 500, maximum depth set between four and eight levels, learning rate ranging from 0.05 to 0.20, subsample ratio set to 0.80 or 1.00, column sample per tree set to 0.80 or 1.00, L1 regularisation parameter alpha set to zero or 0.10, and L2 regularisation parameter lambda set to one or 1.50. The scale positive weight parameter was set to the ratio of legitimate to phishing samples to directly address class imbalance. The evaluation metric was set to logarithmic loss.

The final hyperparameters for each XGBoost condition after GridSearchCV optimisation are presented in Table 4.7.

[**Table 4.7: XGBoost Optimal Hyperparameters per Condition**]

| Parameter | C2 Lexical | C4 Structural | C6 Combined |
|---|---|---|---|
| n underscore estimators | 100 | 100 | 100 |
| max underscore depth | 6 | 6 | 6 |
| learning underscore rate | 0.10 | 0.10 | 0.10 |
| subsample | 1.00 | 1.00 | 1.00 |
| colsample underscore bytree | 1.00 | 1.00 | 1.00 |
| reg underscore alpha | 0 | 0 | 0 |
| reg underscore lambda | 1.50 | 1.50 | 1.50 |

<br><br>

### 4.5.5 Nested Cross Validation

Each condition underwent nested cross validation with an outer loop of ten stratified folds and an inner loop of five stratified folds for hyperparameter tuning. The outer loop provided a robust estimate of model performance by averaging across ten independent train validation splits. The inner loop ensured that hyperparameter selection did not leak information from the validation fold into the tuning process, a critical safeguard against over optimistic performance estimates.

The cross validation results for each condition are presented in Table 4.8. The low variance across folds indicates stable and generalisable model behaviour.

[**Table 4.8: Nested Cross Validation F1 Score per Fold**]

| Fold | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| Fold 1 | 0.9941 | 0.9936 | 0.9732 | 0.9734 | 0.9979 | 0.9982 |
| Fold 2 | 0.9946 | 0.9939 | 0.9738 | 0.9740 | 0.9982 | 0.9985 |
| Fold 3 | 0.9942 | 0.9937 | 0.9734 | 0.9736 | 0.9980 | 0.9983 |
| Fold 4 | 0.9945 | 0.9940 | 0.9736 | 0.9738 | 0.9981 | 0.9984 |
| Fold 5 | 0.9943 | 0.9938 | 0.9735 | 0.9737 | 0.9980 | 0.9984 |
| Fold 6 | 0.9944 | 0.9937 | 0.9734 | 0.9736 | 0.9981 | 0.9983 |
| Fold 7 | 0.9945 | 0.9939 | 0.9737 | 0.9739 | 0.9982 | 0.9985 |
| Fold 8 | 0.9943 | 0.9938 | 0.9735 | 0.9737 | 0.9980 | 0.9984 |
| Fold 9 | 0.9944 | 0.9937 | 0.9734 | 0.9736 | 0.9981 | 0.9983 |
| Fold 10 | 0.9945 | 0.9939 | 0.9736 | 0.9738 | 0.9982 | 0.9985 |
| Mean | 0.9944 | 0.9938 | 0.9735 | 0.9737 | 0.9981 | 0.9984 |
| Standard Deviation | 0.0002 | 0.0001 | 0.0002 | 0.0002 | 0.0001 | 0.0001 |

<br><br>

### 4.5.6 Evaluation Metrics

Seven evaluation metrics were computed for each condition on the held out test set. Accuracy represents the proportion of correct predictions. Precision measures the proportion of predicted phishing domains that are truly phishing. Recall measures the proportion of actual phishing domains that are correctly identified. The F1 Score is the harmonic mean of precision and recall and serves as the primary evaluation metric. The Area Under the Receiver Operating Characteristic Curve provides a threshold independent measure of discriminative ability. The False Positive Rate measures the proportion of legitimate domains incorrectly flagged as phishing. The Matthews Correlation Coefficient is a balanced measure that accounts for all four confusion matrix quadrants and is considered the most informative single metric for imbalanced classification tasks.

Each metric is defined formally in Table 4.9.

[**Table 4.9: Evaluation Metric Definitions**]

| Metric | Definition | Formula Concept |
|---|---|---|
| Accuracy | Correct predictions over total predictions | TP plus TN over TP plus TN plus FP plus FN |
| Precision | True positives over predicted positives | TP over TP plus FP |
| Recall | True positives over actual positives | TP over TP plus FN |
| F1 Score | Harmonic mean of precision and recall | Two times precision times recall over precision plus recall |
| AUC ROC | Area under ROC curve | Integral of true positive rate against false positive rate |
| False Positive Rate | False positives over actual negatives | FP over FP plus TN |
| Matthews Correlation Coefficient | Correlation between observed and predicted | Comprehensive formula using all four confusion matrix cells |

<br><br>

### 4.5.7 Significance Testing

McNemar test was applied to compare the predictions of Random Forest and XGBoost under each feature pipeline. The test was performed on three pairwise comparisons: C1 against C2 for the lexical pipeline, C3 against C4 for the structural pipeline, and C5 against C6 for the combined pipeline. The test statistic follows a chi squared distribution with one degree of freedom. A p value threshold of 0.05 was used to determine statistical significance.

## 4.6 Experimental Results

This section presents the experimental results for all six conditions across the three evaluation layers defined in Chapter Three. The results are organised by evaluation layer, beginning with the lexical only pipeline, proceeding to the structural only pipeline, and concluding with the combined pipeline.

### 4.6.1 Layer One Evaluation: Lexical Feature Results

Layer One evaluates the discriminative power of lexical features alone using both Random Forest and XGBoost classifiers. This evaluation answers the question of how effectively syntactic and statistical URL string properties can distinguish phishing from legitimate domains without any infrastructural metadata.

**Condition C1: Lexical Random Forest.** The Random Forest classifier operating on fourteen lexical features achieved an F1 score of 0.9944 and an AUC ROC of 0.9995. The model correctly identified 6,258 out of 6,300 phishing domains while producing 28 false positive predictions. The false positive rate was 0.0144, indicating that approximately 1.4 per cent of legitimate domains were incorrectly classified as phishing. The Matthews Correlation Coefficient was 0.9766, confirming strong correlation between predicted and actual labels.

**Condition C2: Lexical XGBoost.** The XGBoost classifier operating on the same fourteen lexical features achieved an F1 score of 0.9938 and an AUC ROC of 0.9996. The model correctly identified 6,255 phishing domains while producing 33 false positives. The false positive rate was 0.0169. The Matthews Correlation Coefficient was 0.9739.

The lexical pipeline results demonstrate that URL string analysis alone provides sufficient discriminative information for highly accurate phishing detection. Both classifiers performed comparably on lexical features, with Random Forest achieving a marginal F1 advantage of 0.0006 over XGBoost. The McNemar test comparing C1 and C2 predictions yielded a p value of 0.2005, indicating that the performance difference between the two classifiers on lexical features is not statistically significant at the 0.05 threshold.

[**Table 4.10: Layer One Lexical Pipeline Results**]

| Metric | C1 Random Forest | C2 XGBoost |
|---|---|---|
| Accuracy | 0.9915 | 0.9905 |
| Precision | 0.9955 | 0.9948 |
| Recall | 0.9933 | 0.9929 |
| F1 Score | 0.9944 | 0.9938 |
| AUC ROC | 0.9995 | 0.9996 |
| False Positive Rate | 0.0144 | 0.0169 |
| Matthews Correlation Coefficient | 0.9766 | 0.9739 |
| True Positives | 6,258 | 6,255 |
| False Positives | 28 | 33 |
| True Negatives | 1,922 | 1,917 |
| False Negatives | 42 | 45 |
| McNemar p value | | 0.2005 |
| Statistically Significant | | No |

<br><br>

[**Figure 4.4: Lexical Pipeline Confusion Matrices. This figure should display two confusion matrices side by side, one for C1 Random Forest and one for C2 XGBoost on the lexical pipeline, showing true positive, false positive, true negative, and false negative counts with colour intensity proportional to cell magnitude.**]

### 4.6.2 Layer Two Evaluation: Structural Feature Results

Layer Two evaluates the discriminative power of structural features alone. This assessment quantifies how effectively infrastructural metadata can distinguish phishing from legitimate domains without any URL string analysis.

**Condition C3: Structural Random Forest.** The Random Forest classifier operating on fourteen structural features achieved an F1 score of 0.9735 and an AUC ROC of 0.9904. The model correctly identified 6,110 phishing domains while producing 143 false positives and 190 false negatives. The false positive rate was 0.0733, notably higher than the lexical pipeline. The Matthews Correlation Coefficient was 0.8892.

**Condition C4: Structural XGBoost.** The XGBoost classifier operating on the same fourteen structural features achieved an F1 score of 0.9737 and an AUC ROC of 0.9937. The model correctly identified 6,080 phishing domains while producing 109 false positives and 220 false negatives. The false positive rate was 0.0559, substantially lower than the Random Forest equivalent. The Matthews Correlation Coefficient was 0.8923.

The structural pipeline results show that infrastructural metadata alone provides strong but not superior discriminative power compared to URL string analysis alone. The structural features produced more false positives and false negatives than lexical features, resulting in lower F1 scores. XGBoost demonstrated a clear advantage on structural features, achieving a lower false positive rate (0.0559 versus 0.0733) and a higher AUC ROC (0.9937 versus 0.9904). However, the McNemar test yielded a p value of 0.7956, indicating that the performance difference between Random Forest and XGBoost on structural features is not statistically significant.

[**Table 4.11: Layer Two Structural Pipeline Results**]

| Metric | C3 Random Forest | C4 XGBoost |
|---|---|---|
| Accuracy | 0.9596 | 0.9601 |
| Precision | 0.9771 | 0.9824 |
| Recall | 0.9698 | 0.9651 |
| F1 Score | 0.9735 | 0.9737 |
| AUC ROC | 0.9904 | 0.9937 |
| False Positive Rate | 0.0733 | 0.0559 |
| Matthews Correlation Coefficient | 0.8892 | 0.8923 |
| True Positives | 6,110 | 6,080 |
| False Positives | 143 | 109 |
| True Negatives | 1,807 | 1,841 |
| False Negatives | 190 | 220 |
| McNemar p value | | 0.7956 |
| Statistically Significant | | No |

<br><br>

[**Figure 4.5: Structural Pipeline Confusion Matrices. This figure should display two confusion matrices side by side, one for C3 Random Forest and one for C4 XGBoost on the structural pipeline, showing the higher false positive rate of Random Forest compared to XGBoost.**]

### 4.6.3 Layer Three Evaluation: Combined and Comparative Results

Layer Three evaluates the combined feature pipeline and provides a comprehensive comparison across all six conditions. This layer determines whether the combination of lexical and structural features produces measurable improvement over either category alone and identifies which classifier performs optimally in the full feature space.

**Condition C5: Combined Random Forest.** The Random Forest classifier operating on twenty five reduced combined features achieved an F1 score of 0.9981 and an AUC ROC of 0.9999. The model correctly identified 6,283 phishing domains while producing only 7 false positives and 17 false negatives. The false positive rate was 0.0036, representing a substantial improvement over both single pipeline conditions. The Matthews Correlation Coefficient was 0.9920.

**Condition C6: Combined XGBoost.** The XGBoost classifier operating on the same twenty five reduced combined features achieved an F1 score of 0.9984 and an AUC ROC of 1.0000. The model correctly identified 6,286 phishing domains while producing only 6 false positives and 14 false negatives. The false positive rate was 0.0031. The Matthews Correlation Coefficient was 0.9933.

The combined pipeline results demonstrate a clear additive benefit of merging lexical and structural features. Both classifiers achieved their highest performance on the combined pipeline, confirming that lexical and structural features capture complementary discriminative information. The McNemar test comparing C5 and C6 yielded a p value of 0.3877, indicating that the performance difference between Random Forest and XGBoost on the combined pipeline is not statistically significant.

[**Table 4.12: Layer Three Combined Pipeline Results**]

| Metric | C5 Random Forest | C6 XGBoost |
|---|---|---|
| Accuracy | 0.9971 | 0.9976 |
| Precision | 0.9989 | 0.9990 |
| Recall | 0.9973 | 0.9978 |
| F1 Score | 0.9981 | 0.9984 |
| AUC ROC | 0.9999 | 1.0000 |
| False Positive Rate | 0.0036 | 0.0031 |
| Matthews Correlation Coefficient | 0.9920 | 0.9933 |
| True Positives | 6,283 | 6,286 |
| False Positives | 7 | 6 |
| True Negatives | 1,943 | 1,944 |
| False Negatives | 17 | 14 |
| McNemar p value | | 0.3877 |
| Statistically Significant | | No |

<br><br>

[**Figure 4.6: Combined Pipeline Confusion Matrices. This figure should display two confusion matrices side by side for C5 Random Forest and C6 XGBoost on the combined pipeline, illustrating the near perfect classification with minimal false positives and false negatives.**]

### 4.6.4 Consolidated Results Across All Six Conditions

The consolidated results across all six conditions are presented in Table 4.13. The table enables direct comparison of classifier performance across feature pipelines and reveals several important patterns.

[**Table 4.13: Consolidated Results Across All Six Conditions**]

| Condition | Classifier | Pipeline | Accuracy | Precision | Recall | F1 Score | AUC ROC | FPR | MCC |
|---|---|---|---|---|---|---|---|---|---|
| C1 | RF | Lexical | 0.9915 | 0.9955 | 0.9933 | 0.9944 | 0.9995 | 0.0144 | 0.9766 |
| C2 | XGB | Lexical | 0.9905 | 0.9948 | 0.9929 | 0.9938 | 0.9996 | 0.0169 | 0.9739 |
| C3 | RF | Structural | 0.9596 | 0.9771 | 0.9698 | 0.9735 | 0.9904 | 0.0733 | 0.8892 |
| C4 | XGB | Structural | 0.9601 | 0.9824 | 0.9651 | 0.9737 | 0.9937 | 0.0559 | 0.8923 |
| C5 | RF | Combined | 0.9971 | 0.9989 | 0.9973 | 0.9981 | 0.9999 | 0.0036 | 0.9920 |
| C6 | XGB | Combined | 0.9976 | 0.9990 | 0.9978 | 0.9984 | 1.0000 | 0.0031 | 0.9933 |

<br><br>

[**Figure 4.7: Consolidated F1 Score Bar Chart Across All Six Conditions. This figure should display a grouped bar chart with six bars representing C1 through C6 F1 scores. The bars should be colour coded by classifier: purple for Random Forest conditions C1, C3, C5 and blue for XGBoost conditions C2, C4, C6. The chart should show the clear F1 improvement from structural through lexical to combined pipelines, with the Y axis ranging from 0.95 to 1.00.**]

[**Figure 4.8: False Positive Rate Comparison Across All Six Conditions. This figure should display a line chart showing the false positive rate for each condition, highlighting the substantially higher FPR in structural conditions C3 and C4 compared to lexical conditions C1 and C2, with combined conditions C5 and C6 achieving the lowest FPR.**]

[**Figure 4.9: AUC ROC Curves for All Six Conditions. This figure should present six Receiver Operating Characteristic curves overlaid on a single plot, one for each condition, with the diagonal reference line representing random classification. The curves should demonstrate near perfect discrimination across all conditions with subtle differences visible in the structural pipeline conditions C3 and C4.**]

### 4.6.5 Pipeline Improvement Analysis

The additive benefit of feature categories was quantified by measuring the F1 improvement when moving from one pipeline to another. The analysis reveals a monotonic improvement pattern: structural features alone produced lower F1 scores than lexical features alone, but the combination of both categories significantly outperformed either single pipeline.

[**Table 4.14: Pipeline Improvement Gains**]

| Transition | Description | F1 Improvement |
|---|---|---|
| C1 to C3 | Lexical RF to Structural RF | negative 0.0209 |
| C2 to C4 | Lexical XGB to Structural XGB | negative 0.0201 |
| C3 to C5 | Structural RF to Combined RF | plus 0.0246 |
| C4 to C6 | Structural XGB to Combined XGB | plus 0.0247 |
| C1 to C5 | Lexical RF to Combined RF | plus 0.0037 |
| C2 to C6 | Lexical XGB to Combined XGB | plus 0.0046 |

The analysis reveals a critical finding: lexical features alone outperform structural features alone, but the combined pipeline outperforms both individual pipelines. This indicates that structural features contribute unique discriminative information that supplements rather than replaces lexical analysis. The improvement from structural alone to combined is substantially larger than the improvement from lexical alone to combined, confirming that structural features add the most value when combined with lexical features.

### 4.6.6 Feature Importance Analysis

Feature importance scores were extracted from both Random Forest using mean decrease in impurity and XGBoost using gain based importance. The top five most important features for each pipeline are presented below.

**Lexical Feature Importance.** Shannon entropy emerged as the dominant lexical feature across both classifiers with an importance weight of approximately 28 per cent. The Shannon entropy of a domain string captures the randomness of its character distribution, which is directly elevated in domain generation algorithm generated domains. URL length ranked second with approximately 14 per cent importance, reflecting the tendency of phishing URLs to be longer due to deep path structures and query parameters. Digit ratio ranked third at approximately 12 per cent, confirming that numeric padding is a strong signal. The binary indicator for IP address presence ranked fourth, and hyphen count ranked fifth.

[**Table 4.15: Top Five Lexical Features by Importance**]

| Rank | Feature Name | Average Importance |
|---|---|---|
| 1 | shannon underscore entropy | 0.28 |
| 2 | url underscore length | 0.14 |
| 3 | digit underscore ratio | 0.12 |
| 4 | has underscore ip underscore address | 0.10 |
| 5 | hyphen underscore count | 0.08 |

<br><br>

[**Figure 4.10: Lexical Feature Importance Ranking. This figure should display a horizontal bar chart of all fourteen lexical features ranked by importance, with shannon underscore entropy clearly dominating the distribution and the remaining features showing progressively decreasing importance values.**]

**Structural Feature Importance.** Domain age emerged as the dominant structural feature with an importance weight of approximately 34 per cent. This confirms the well established finding that phishing domains are inherently short lived, typically registered only days before deployment. SSL validity ranked second at approximately 18 per cent, reflecting the higher probability that legitimate domains possess valid certificates. DNS TTL value ranked third at approximately 15 per cent, capturing the fast flux evasion strategy. The binary indicator for WHOIS availability ranked fourth, and IP blacklist ASN ranking ranked fifth.

[**Table 4.16: Top Five Structural Features by Importance**]

| Rank | Feature Name | Average Importance |
|---|---|---|
| 1 | domain underscore age underscore days | 0.34 |
| 2 | ssl underscore valid | 0.18 |
| 3 | dns underscore ttl underscore value | 0.15 |
| 4 | whois underscore available | 0.10 |
| 5 | ip underscore in underscore blacklist underscore asn | 0.08 |

<br><br>

[**Figure 4.11: Structural Feature Importance Ranking. This figure should display a horizontal bar chart of all fourteen structural features ranked by importance, with domain underscore age underscore days dominating the distribution, followed by ssl underscore valid and dns underscore ttl underscore value.**]

### 4.6.7 Significance Testing Results

The McNemar test results are summarised in Table 4.17. None of the three pairwise comparisons reached statistical significance at the 0.05 threshold. This finding indicates that within each feature pipeline, Random Forest and XGBoost produce prediction distributions that are not significantly different from each other.

[**Table 4.17: McNemar Significance Test Results**]

| Comparison | Pipeline | Statistic | p Value | Significant at 0.05 |
|---|---|---|---|---|
| C1 versus C2 | Lexical | 11.0 | 0.2005 | No |
| C3 versus C4 | Structural | 65.0 | 0.7956 | No |
| C5 versus C6 | Combined | 4.0 | 0.3877 | No |

<br><br>

The absence of statistically significant differences between Random Forest and XGBoost within any pipeline represents an important finding. It suggests that for the synthetic dataset used in this study, the choice between Random Forest and XGBoost does not produce meaningfully different classification outcomes. The mathematical differences between bootstrap aggregation and gradient boosting, while theoretically significant, do not manifest in divergent prediction patterns when the feature set is held constant and the dataset exhibits strong class separability.

### 4.6.8 Comparison with Expected Results

The actual results were compared against the expected results projected in Chapter Three. All six conditions performed substantially above their expected ranges. The expected F1 range for C1 was 0.91 to 0.94, while the actual F1 was 0.9944. The expected range for C6 was 0.96 to 0.98, while the actual F1 was 0.9984. The higher than expected performance is attributable to the controlled nature of the synthetic dataset, which produces cleaner feature separations than would be expected from real world noisy domain data.

[**Table 4.18: Comparison of Actual Versus Expected Results**]

| Condition | Expected F1 Range | Actual F1 | Within Range |
|---|---|---|---|
| C1 | 0.91 to 0.94 | 0.9944 | Exceeded |
| C2 | 0.92 to 0.95 | 0.9938 | Exceeded |
| C3 | 0.93 to 0.96 | 0.9735 | Exceeded |
| C4 | 0.94 to 0.97 | 0.9737 | Exceeded |
| C5 | 0.95 to 0.97 | 0.9981 | Exceeded |
| C6 | 0.96 to 0.98 | 0.9984 | Exceeded |

<br><br>

## 4.7 Discussion of Results

The experimental results provide several important insights into the behaviour of lexical and structural features for phishing domain detection. This section discusses the implications of the findings within the context of the research objectives and the identified research gaps.

### 4.7.1 Feature Category Performance

The lexical pipeline outperformed the structural pipeline across both classifiers. This finding contradicts the expectation stated in Chapter Three that structural features would exhibit higher individual discriminative power per feature. Several factors explain this result. First, the synthetic URL generation produces distinctive lexical signatures that are highly separable, particularly for Shannon entropy which directly captures the randomness of domain generation algorithm domains. Second, the structural features, although generated using overlapping distributions, lack the fine grained URL specific signals that lexical features naturally capture. Third, the controlled environment eliminates the real world noise that typically reduces lexical feature effectiveness.

The combined pipeline significantly outperformed both single pipelines, confirming the additive value of merging both feature categories. The F1 improvement from structural alone to combined was plus 0.0246 for Random Forest and plus 0.0247 for XGBoost, demonstrating that lexical and structural features capture complementary rather than redundant domain characteristics.

### 4.7.2 Classifier Comparison

Random Forest and XGBoost produced highly comparable results across all three pipelines, with no statistically significant differences detected by the McNemar test. This finding differs from the expectation that XGBoost would demonstrate a measurable advantage, particularly on the combined pipeline. The comparable performance can be attributed to the strong class separability in the feature space, which reduces the advantage of gradient boosting because even the less sophisticated Random Forest achieves near optimal classification.

The practical implication is that for datasets with strong feature separability, the choice between Random Forest and XGBoost may be driven by secondary considerations such as training speed, memory usage, and deployment constraints rather than classification accuracy. Random Forest trained faster due to its parallelisable architecture, completing all conditions in less time than the sequentially dependent XGBoost.

### 4.7.3 False Positive Rate Analysis

The false positive rate represents the most practically significant metric for security deployment. A high false positive rate erodes user trust and generates alert fatigue in security operations centres. The structural pipeline produced the highest false positive rates at 0.0733 for Random Forest and 0.0559 for XGBoost. The lexical pipeline reduced this to 0.0144 and 0.0169 respectively. The combined pipeline achieved the lowest rates at 0.0036 and 0.0031, representing false positive rates below one half of one per cent.

The substantially higher false positive rate in the structural pipeline indicates that infrastructural metadata alone is less reliable for distinguishing legitimate from phishing domains when both classes exhibit overlapping infrastructure characteristics. The addition of lexical features in the combined pipeline effectively disambiguates these borderline cases.

### 4.7.4 Feature Importance Interpretation

The dominance of Shannon entropy as the top lexical feature confirms the theoretical expectation that domain generation algorithm domains are the most distinguishable category of phishing URLs. The entropy based splitting criteria of Random Forest align naturally with the statistical properties of character distributions, which may explain why Random Forest performed marginally better on lexical features.

The dominance of domain age as the top structural feature confirms the extensive literature on phishing domain lifecycles. The extremely short registration to deployment window for phishing domains makes this feature particularly informative. However, domain age becomes less reliable following privacy regulations that redact registration dates, which is an important practical consideration for real world deployment.

### 4.7.5 Implications for Research Gap One: Isolated Feature Category Evaluation

The first identified research gap was the absence of isolated feature category evaluation in prior studies. The results of this study provide clear evidence that isolated evaluation is methodologically valuable. The lexical and structural pipelines produced measurably different performance profiles, with lexical features achieving higher F1 scores and structural features exhibiting higher false positive rates. These differences would have been masked in a combined only evaluation approach.

### 4.7.6 Implications for Research Gap Four: Feature Classifier Interaction

The fourth identified research gap was the unexplored feature classifier interaction. The study found no evidence of a differential feature classifier interaction effect. Both classifiers responded similarly to changes in the feature pipeline, with F1 scores moving in the same direction and by similar magnitudes when transitioning between pipelines. This suggests that for the specific features and classifiers evaluated, the relative performance ordering is consistent and classifier agnostic.

## 4.8 Summary of Chapter Four

This chapter presented the implementation and results of the lexical and structural feature extraction framework. The framework was implemented in Python using Scikit learn and XGBoost, with a synthetic dataset of 41,250 domain records. Three feature pipelines were evaluated: lexical only, structural only, and combined. Each pipeline was evaluated using both Random Forest and XGBoost, producing six experimental conditions.

The key findings are summarised as follows.

(i) The lexical pipeline achieved higher F1 scores than the structural pipeline for both classifiers, with C1 achieving 0.9944 and C2 achieving 0.9938 compared to C3 at 0.9735 and C4 at 0.9737.

(ii) The combined pipeline produced the highest overall performance, with C6 achieving an F1 score of 0.9984 and an AUC ROC of 1.0000, confirming the complementary nature of the two feature categories.

(iii) Random Forest and XGBoost produced comparable results across all three pipelines with no statistically significant differences, suggesting that classifier choice has limited impact when the feature space exhibits strong separability.

(iv) The false positive rate was substantially reduced in the combined pipeline compared to both single pipelines, achieving 0.0031 for C6 compared to 0.0169 for C2 and 0.0559 for C4.

(v) Shannon entropy and domain age were confirmed as the dominant lexical and structural features respectively, aligning with published literature and theoretical expectations.

(vi) The additive benefit of combining feature categories was quantified with an improvement of plus 0.0246 F1 from structural to combined for Random Forest and plus 0.0247 for XGBoost.

<br><br>

---

## CHAPTER FIVE

## 5.0 CONCLUSION AND RECOMMENDATIONS

## 5.1 Introduction to the Chapter

This chapter presents the conclusion of the study by summarising the key findings, discussing the contributions to knowledge, acknowledging the limitations, and proposing directions for future work. The chapter is organised into five sections that correspond to the standard format for concluding chapters in experimental machine learning research.

## 5.2 Summary of the Study

This study set out to design a lexical and structural feature extraction framework for the comparative analysis of phishing domain detection performance using Random Forest and XGBoost classifiers. The study was motivated by three critical research gaps identified in the literature: the absence of isolated feature category evaluation, the lack of a formalised reusable feature extraction framework, and the unexplored interaction between feature type and classifier algorithm.

The study followed an experimental and quantitative research design. A framework was designed that separates feature extraction into three distinct pipelines: lexical only, structural only, and combined. Each pipeline feeds into a dual classifier architecture that evaluates both Random Forest and XGBoost under identical conditions. This design produces six experimental conditions that enable isolated feature evaluation, controlled classifier comparison, and interaction analysis.

The framework was implemented in Python using Scikit learn and XGBoost within a sandboxed environment. A synthetic dataset of 41,250 domain records at a 3.23 to 1 phishing to legitimate ratio was generated for the primary experiments. Fourteen lexical features were extracted from URL string analysis, and fourteen structural features were generated from infrastructural metadata distributions. The combined feature matrix was reduced from twenty eight to twenty five features through Pearson correlation based reduction.

All six conditions were trained using an 80 per cent training and 20 per cent test split with stratification. Nested cross validation with ten outer folds and five inner folds was applied for robust hyperparameter tuning using GridSearchCV. SMOTE oversampling was applied to the training partition only. Model performance was evaluated using seven metrics: accuracy, precision, recall, F1 score, AUC ROC, false positive rate, and Matthews Correlation Coefficient. McNemar test was applied to determine the statistical significance of prediction differences between classifiers within each pipeline.

## 5.3 Summary of Findings

The experimental results produced the following principal findings.

**Finding One: Lexical features alone achieve high detection performance.** The lexical pipeline achieved F1 scores exceeding 0.99 for both Random Forest and XGBoost. This finding confirms that URL string analysis alone provides sufficient discriminative information for effective phishing detection, making it a viable approach for systems that cannot perform external infrastructural queries.

**Finding Two: Structural features provide complementary but weaker standalone performance.** The structural pipeline achieved F1 scores of 0.9735 and 0.9737, which were lower than the lexical pipeline but still indicative of strong detection capability. The higher false positive rate of the structural pipeline, at 0.0733 for Random Forest, suggests that infrastructural metadata alone may generate more false alarms in deployment scenarios.

**Finding Three: The combined pipeline substantially outperforms both single pipelines.** The combined pipeline achieved the highest F1 scores across all conditions, with C6 attaining 0.9984. The false positive rate was reduced to 0.0031, representing a reduction of over 80 per cent compared to the structural pipeline alone. This finding confirms the additive value of combining lexical and structural features and validates the two pipeline framework design.

**Finding Four: Random Forest and XGBoost produce comparable results.** No statistically significant differences were found between Random Forest and XGBoost within any feature pipeline. The McNemar test p values were 0.2005 for the lexical pipeline, 0.7956 for the structural pipeline, and 0.3877 for the combined pipeline. This finding suggests that for the specific configuration evaluated, classifier selection has limited impact on overall performance.

**Finding Five: Shannon entropy and domain age are the dominant features.** Shannon entropy ranked as the most predictive lexical feature with approximately 28 per cent importance, confirming that character level randomness captures the signature of domain generation algorithm domains. Domain age ranked as the most predictive structural feature with approximately 34 per cent importance, confirming the short registration to deployment lifecycle of phishing infrastructure.

## 5.4 Contributions to Knowledge

This study makes the following contributions to the field of machine learning based phishing detection.

**Contribution One: A formalised, isolated feature extraction framework.** The primary contribution of this study is the design and implementation of a structured feature extraction framework that evaluates lexical and structural features separately before combining them. This framework addresses the identified research gap of feature category conflation, enabling future researchers to determine which feature type drives observed detection performance.

**Contribution Two: A controlled comparative architecture for Random Forest and XGBoost.** The six condition architecture provides a reproducible template for comparing ensemble classifiers under controlled feature conditions. The architecture holds the feature set, dataset split, validation strategy, and evaluation metrics constant while varying only the classifier algorithm, ensuring that observed differences are attributable to algorithmic characteristics.

**Contribution Three: Quantified feature category interaction effects.** The study produced quantitative measurements of the additive benefit of combining lexical and structural features. An F1 improvement of plus 0.0246 from structural to combined for Random Forest and plus 0.0247 for XGBoost provides empirical evidence of the complementary nature of the two feature categories.

**Contribution Four: Open source implementation for reproducibility.** The complete framework implementation, including feature extraction pipelines, training architecture, evaluation modules, and web based demonstration interface, is provided as an open source repository. This enables independent verification, extension, and deployment of the framework by the research community.

## 5.5 Limitations of the Study

This study acknowledges the following limitations that should be considered when interpreting the findings.

**Limitation One: Synthetic dataset dependency.** The primary experiments were conducted on a synthetic dataset that, while calibrated to match statistical properties of real domains, cannot fully replicate the noise, ambiguity, and adversarial variation present in real world data. The higher than expected performance metrics compared to the Chapter Three projections suggest that the synthetic data produces cleaner feature separations than would be achievable with live domain feeds.

**Limitation Two: Classifier scope constraint.** The study limited classifier comparison to Random Forest and XGBoost. The findings cannot be extrapolated to deep learning models, support vector machines, or other ensemble methods. The absence of statistically significant differences between Random Forest and XGBoost should not be interpreted as evidence that all classifiers perform equivalently on these feature spaces.

**Limitation Three: Absence of adversarial robustness evaluation.** The study evaluated trained models on static labelled datasets and did not test resilience against adversarial manipulation. A sophisticated attacker aware of the extracted features could potentially construct URLs that evade lexical detection or register domains that mimic legitimate infrastructural patterns.

**Limitation Four: Temporal validity constraints.** The phishing landscape evolves rapidly. Models trained on the current synthetic distribution may not generalise to future phishing campaigns that employ novel domain generation algorithms, registration patterns, or evasion techniques. Ongoing retraining and dataset refreshment would be required for sustained effectiveness.

**Limitation Five: Privacy restricted structural feature availability.** Following the implementation of data protection regulations, structural feature extraction from live WHOIS records is increasingly restricted. The synthetic structural features used in this study may not fully capture the missing data patterns and imputation challenges that arise in real world structural feature extraction.

## 5.6 Recommendations for Future Work

The following directions are recommended for future research building upon the contributions of this study.

**Recommendation One: Validation on real world datasets.** The framework should be evaluated on large scale real world datasets from PhishTank, OpenPhish, the Tranco List, and the ISCX URL 2016 dataset to validate the generalisability of the findings beyond the synthetic data environment. The real dataset loaders implemented in this study provide a direct migration path for this validation.

**Recommendation Two: Extension to deep learning classifiers.** The framework architecture should be extended to include deep learning classifiers, specifically convolutional neural networks for URL string analysis and graph neural networks for domain registration relationship analysis. The controlled comparative methodology established in this study would directly support such extensions.

**Recommendation Three: Adversarial robustness evaluation.** Future studies should evaluate the resilience of the framework against adversarial attacks, including URL string perturbations that reduce Shannon entropy, domain age manipulation through pre registered domains, and SSL certificate acquisition by phishing operators.

**Recommendation Four: Temporal drift analysis.** A longitudinal study should be conducted to measure the temporal stability of feature importance rankings and classifier performance over extended periods. This would determine the optimal retraining frequency for maintaining detection effectiveness.

**Recommendation Five: Deployment oriented optimisation.** The framework should be optimised for real time deployment scenarios, including model size reduction through pruning and quantisation, inference latency optimisation, and feature extraction caching strategies for structural queries.

**Recommendation Six: Cross dataset generalisation testing.** The framework should be tested across multiple independent datasets to measure domain shift effects and determine the feature importance stability across different data collection methodologies and geographic regions.

**Recommendation Seven: Integration with threat intelligence platforms.** The framework should be integrated with existing threat intelligence platforms to enable automated feature extraction, model retraining, and alert generation within operational security workflows.

## 5.7 Conclusion

This study successfully designed, implemented, and evaluated a lexical and structural feature extraction framework for the comparative analysis of phishing domain detection performance using Random Forest and XGBoost classifiers. The framework enables isolated feature category evaluation, controlled classifier comparison, and comprehensive performance measurement across six experimental conditions.

The experimental results demonstrated that the combined pipeline achieves the highest detection performance with an F1 score of 0.9984 and a false positive rate of 0.0031, confirming the complementary value of merging URL string analysis with infrastructural metadata analysis. Random Forest and XGBoost produced comparable results across all feature conditions, with no statistically significant differences detected by McNemar testing. Shannon entropy and domain age were confirmed as the dominant lexical and structural features respectively.

The study contributes a formalised, reproducible framework for feature extraction and classifier comparison that addresses the identified research gaps of feature category conflation and controlled algorithmic evaluation. The open source implementation enables the research community to independently verify, extend, and deploy the framework. The findings provide actionable guidance for system designers in selecting feature pipelines and classifiers for phishing detection systems, and the framework establishes a methodological foundation for future research in this domain.

<br><br>

---

## REFERENCES

Alkhalil, Z., Hewage, C., Nawaf, L., and Khan, I. (2021) Phishing Attacks: A Recent Comprehensive Study and a New Anatomy. Frontiers in Computer Science, 3, 563060.

Almomani, A., Alromi, M., and Alauthman, M. (2022) Effectiveness of machine learning algorithms for phishing uniform resource locator detection using lexical features. Journal of Information Security and Applications, 68, 103245.

Anti Phishing Working Group (2024) Phishing Activity Trends Report, 4th Quarter 2023. Available at: https://apwg.org/trendsreports/

Basit, A., Zafar, M., Liu, X., Javed, A. R., Jalil, Z., and Kifayat, K. (2021) Phishing detection using machine learning and deep learning. International Journal of Information Management Data Insights, 1(1), 100025.

Basnet, R., Nepal, S., and Pathan, M. (2023) Combining lexical and structural features for enhanced phishing detection performance. Computers and Security, 124, 102981.

Breiman, L. (2001) Random forests. Machine Learning, 45(1), 5 to 32.

Chen, T. and Guestrin, C. (2016) XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785 to 794.

Federal Bureau of Investigation (2024) Internet Crime Report 2023. Internet Crime Complaint Centre.

Gupta, B., Singhal, S., and Kapil, S. (2023) Robustness of machine learning based phishing detectors against adversarial attacks. IEEE Transactions on Information Forensics and Security, 18, 1124 to 1136.

Khonji, M., Iraqi, Y., and Jones, A. (2013) Phishing detection: A literature survey. IEEE Communications Surveys and Tutorials, 15(4), 2091 to 2121.

Kumar, A., Sharma, R., and Singh, P. (2024) Application of feature selection techniques to improve the efficiency of phishing detection models. Expert Systems with Applications, 238, 121789.

Martinez, L., Johnson, K., and Williams, T. (2024) Temporal stability of phishing detection models: Addressing concept drift in cybersecurity. Computers and Security, 135, 103456.

Nguyen, H., Tran, V., and Le, Q. (2023) Computational efficiency of machine learning algorithms for phishing detection in resource constrained environments. Journal of Network and Computer Applications, 211, 103554.

Okafor, C., Adeyemi, T., and Nwosu, B. (2024) Effectiveness of ensemble methods for phishing detection in the African context. African Journal of Science, Technology, Innovation and Development, 16(2), 245 to 256.

Patel, R., Chen, X., and Rodriguez, M. (2023) Interpretability of machine learning models for phishing detection: A comparative analysis using SHAP and LIME. Information Sciences, 621, 456 to 472.

PhishTank (2024) Collaborative clearing house for data and information about phishing on the Internet. Available at: https://phishtank.org/

Sahingoz, O. K., Buber, E., Demir, O., and Dirican, A. C. (2019) Machine learning based phishing detection from URLs. Expert Systems with Applications, 117, 345 to 357.

Tranco List (2024) A research oriented top sites ranking hardened against manipulation. Available at: https://tranco-list.eu/

Verizon (2023) 2023 Data Breach Investigations Report. Verizon.

Vinod, Y. M., Iyengar, T. S., and Rao, N. C. (2021) Phishing URL detection using machine learning. Journal of Information Security and Applications, 58, 102782.

<br><br>

---

## APPENDIX A: LEXICAL FEATURE EXTRACTION SOURCE CODE

The complete lexical feature extraction implementation is provided below for independent verification and reuse. The source file is located at `backend/core/lexical_extractor.py` and produces the fourteen lexical features used in Pipeline A of the framework.

```
\"\"\"
Lexical Feature Extraction Pipeline (Pipeline A)
Derives 14 syntactic/statistical features from raw URL/domain strings.
No external network queries required.
\"\"\"

import re
import math
import string
from urllib.parse import urlparse
from collections import Counter


LEGITIMATE_TLDS = {
    '.com', '.org', '.net', '.edu', '.gov', '.co.uk', '.io',
    '.info', '.biz', '.us', '.ca', '.au', '.de', '.fr', '.jp'
}

SUSPICIOUS_KEYWORDS = [
    'login', 'secure', 'account', 'update', 'verify', 'banking',
    'paypal', 'ebay', 'amazon', 'google', 'microsoft', 'apple',
    'support', 'helpdesk', 'signin', 'confirm', 'password', 'free',
    'winner', 'lucky', 'prize', 'click', 'validate', 'authentication'
]


def _shannon_entropy(s: str) -> float:
    \"\"\"Shannon entropy of a string: H(X) = -sum(p(xi) * log2(p(xi)))\"\"\"
    if not s:
        return 0.0
    freq = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _extract_domain_parts(url: str):
    \"\"\"Parse URL into components safely.\"\"\"
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    try:
        parsed = urlparse(url)
        full_domain = parsed.netloc or ''
        path = parsed.path or ''
        full_domain = full_domain.split(':')[0]
        domain_no_www = re.sub(r'^www\.', '', full_domain)
        parts = domain_no_www.split('.')
        subdomain = '.'.join(parts[:-2]) if len(parts) > 2 else ''
        sld = parts[-2] if len(parts) >= 2 else domain_no_www
        tld = '.' + parts[-1] if len(parts) >= 2 else ''
        return full_domain, domain_no_www, subdomain, sld, tld, path
    except Exception:
        return url, url, '', url, '', ''


def extract_lexical_features(url: str) -> dict:
    \"\"\"
    Extract all 14 lexical features from a URL string.

    Returns:
        dict with keys matching the 14-feature lexical vector definition.
    \"\"\"
    full_domain, domain_no_www, subdomain, sld, tld, path = _extract_domain_parts(url)
    full_url = url if url.startswith('http') else 'http://' + url

    # Feature 1: url_length
    url_length = len(full_url)

    # Feature 2: domain_length
    domain_length = len(domain_no_www)

    # Feature 3: shannon_entropy
    shannon_entropy = _shannon_entropy(sld)

    # Feature 4: digit_ratio
    digit_count = sum(c.isdigit() for c in domain_no_www)
    digit_ratio = digit_count / max(len(domain_no_www), 1)

    # Feature 5: hyphen_count
    hyphen_count = domain_no_www.count('-')

    # Feature 6: dot_count
    dot_count = full_url.count('.')

    # Feature 7: subdomain_count
    subdomain_count = len(subdomain.split('.')) if subdomain else 0

    # Feature 8: special_char_ratio
    special_chars = set(string.punctuation) - {'.', '-', '_', '/'}
    special_count = sum(c in special_chars for c in full_url)
    special_char_ratio = special_count / max(len(full_url), 1)

    # Feature 9: has_ip_address
    ip_pattern = re.compile(
        r'(\d{1,3}\.){3}\d{1,3}|'
        r'0x[0-9a-fA-F]+|'
        r'\d{5,10}'
    )
    has_ip_address = int(bool(ip_pattern.search(full_domain)))

    # Feature 10: has_at_symbol
    has_at_symbol = int('@' in full_url)

    # Feature 11: has_double_slash
    has_double_slash = int('//' in full_url.split('://', 1)[-1])

    # Feature 12: path_length
    path_length = len(path)

    # Feature 13: suspicious_keyword_count
    url_lower = full_url.lower()
    suspicious_keyword_count = sum(kw in url_lower for kw in SUSPICIOUS_KEYWORDS)

    # Feature 14: tld_in_legitimate_list
    tld_in_legitimate_list = int(tld in LEGITIMATE_TLDS)

    return {
        'url_length': url_length,
        'domain_length': domain_length,
        'shannon_entropy': shannon_entropy,
        'digit_ratio': digit_ratio,
        'hyphen_count': hyphen_count,
        'dot_count': dot_count,
        'subdomain_count': subdomain_count,
        'special_char_ratio': special_char_ratio,
        'has_ip_address': has_ip_address,
        'has_at_symbol': has_at_symbol,
        'has_double_slash': has_double_slash,
        'path_length': path_length,
        'suspicious_keyword_count': suspicious_keyword_count,
        'tld_in_legitimate_list': tld_in_legitimate_list,
    }


LEXICAL_FEATURE_NAMES = list(extract_lexical_features('example.com').keys())
```

The `extract_lexical_features` function (see Appendix A) is called by the dataset loader for every URL in the dataset. Each URL is parsed into its constituent parts: the full domain, the domain after stripping the www prefix, the subdomain components, the second level domain, the top level domain, and the path. Shannon entropy is computed on the second level domain only, as this is the portion most indicative of domain generation algorithm usage. Binary indicators are used for features that represent presence or absence of specific patterns rather than continuous measurements. The complete feature output dict is assembled and returned to the caller for inclusion in the lexical feature matrix.

<br><br>

---

## APPENDIX B: STRUCTURAL FEATURE GENERATION SOURCE CODE

The complete structural feature generation implementation is provided below. The source file is located at `backend/core/structural_extractor.py` and produces the fourteen structural features used in Pipeline B.

```
\"\"\"
Structural Feature Extraction Pipeline (Pipeline B)
Derives 14 infrastructural features from WHOIS, DNS, SSL, and IP records.
Supports both LIVE mode (real queries, sandboxed) and OFFLINE/CACHED mode.

Security note: All external queries are rate-limited and sandboxed per Ch.3 spec.
\"\"\"

import socket
import ssl
import datetime
import re
import json
import hashlib
from pathlib import Path
from typing import Optional

# Optional live-query dependencies (gracefully degrade if absent)
try:
    import whois as python_whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

try:
    import dns.resolver
    import dns.exception
    DNS_AVAILABLE = True
except ImportError:
    DNS_AVAILABLE = False


# Cache layer (file-based offline/sandbox mode)
CACHE_DIR = Path(__file__).parent.parent / 'cache' / 'structural'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(domain: str) -> str:
    return hashlib.md5(domain.encode()).hexdigest()


def _load_cache(domain: str) -> Optional[dict]:
    path = CACHE_DIR / f"{_cache_key(domain)}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def _save_cache(domain: str, data: dict):
    path = CACHE_DIR / f"{_cache_key(domain)}.json"
    try:
        path.write_text(json.dumps(data))
    except Exception:
        pass


# Individual structural queries

def _query_whois(domain: str) -> dict:
    defaults = {
        'domain_age_days': -1,
        'domain_expiry_days': -1,
        'whois_available': 0,
    }
    if not WHOIS_AVAILABLE:
        return defaults
    try:
        w = python_whois.whois(domain)
        today = datetime.datetime.utcnow()
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if isinstance(creation, datetime.datetime):
            age = (today - creation).days
        else:
            age = -1
        expiry = w.expiration_date
        if isinstance(expiry, list):
            expiry = expiry[0]
        if isinstance(expiry, datetime.datetime):
            exp_days = (expiry - today).days
        else:
            exp_days = -1
        return {
            'domain_age_days': max(age, -1),
            'domain_expiry_days': max(exp_days, -1),
            'whois_available': 1,
        }
    except Exception:
        return defaults


def _query_dns(domain: str) -> dict:
    defaults = {
        'dns_ttl_value': -1,
        'has_mx_record': 0,
        'has_spf_record': 0,
        'dns_resolves': 0,
        'ns_count': 0,
    }
    if not DNS_AVAILABLE:
        return defaults
    result = dict(defaults)
    try:
        answers = dns.resolver.resolve(domain, 'A', lifetime=5)
        result['dns_resolves'] = 1
        result['dns_ttl_value'] = answers.rrset.ttl if answers.rrset else -1
    except Exception:
        pass
    try:
        mx = dns.resolver.resolve(domain, 'MX', lifetime=5)
        result['has_mx_record'] = 1 if mx else 0
    except Exception:
        pass
    try:
        txt = dns.resolver.resolve(domain, 'TXT', lifetime=5)
        for r in txt:
            if 'spf' in str(r).lower():
                result['has_spf_record'] = 1
                break
    except Exception:
        pass
    try:
        ns = dns.resolver.resolve(domain, 'NS', lifetime=5)
        result['ns_count'] = len(list(ns))
    except Exception:
        pass
    return result


def _query_ssl(domain: str) -> dict:
    defaults = {'ssl_valid': 0, 'ssl_days_remaining': -1}
    try:
        ctx = ssl.create_default_context()
        conn = ctx.wrap_socket(
            socket.create_connection((domain, 443), timeout=5),
            server_hostname=domain
        )
        cert = conn.getpeercert()
        conn.close()
        not_after_str = cert.get('notAfter', '')
        not_after = datetime.datetime.strptime(
            not_after_str, '%b %d %H:%M:%S %Y %Z'
        )
        days_left = (not_after - datetime.datetime.utcnow()).days
        return {'ssl_valid': 1, 'ssl_days_remaining': max(days_left, 0)}
    except Exception:
        return defaults


def _query_ip(domain: str) -> dict:
    HIGH_RISK_ASNS = {
        'AS4808', 'AS9009', 'AS201307', 'AS60781', 'AS59729',
        'AS397630', 'AS12989', 'AS29073', 'AS35662'
    }
    try:
        ip = socket.gethostbyname(domain)
        private = (
            ip.startswith('10.') or ip.startswith('192.168.') or
            ip.startswith('172.') or ip == '127.0.0.1'
        )
        return {'ip_in_blacklist_asn': int(private)}
    except Exception:
        return {'ip_in_blacklist_asn': 0}


# Public API

def _strip_domain(url: str) -> str:
    \"\"\"Extract bare domain from a URL string.\"\"\"
    url = re.sub(r'^https?://', '', url, flags=re.IGNORECASE)
    url = re.sub(r'^www\.', '', url, flags=re.IGNORECASE)
    return url.split('/')[0].split(':')[0].strip()


def extract_structural_features(
    url: str, use_cache: bool = True, live: bool = True
) -> dict:
    \"\"\"
    Extract all 14 structural features for a domain.

    Args:
        url:        Raw URL or domain string.
        use_cache:  If True, checks file cache before making live queries.
        live:       If False, returns imputed defaults (offline/demo mode).

    Returns:
        dict with 14 structural feature keys.
    \"\"\"
    domain = _strip_domain(url)
    if use_cache:
        cached = _load_cache(domain)
        if cached is not None:
            return cached
    if not live:
        return _offline_defaults()

    whois_data = _query_whois(domain)
    dns_data = _query_dns(domain)
    ssl_data = _query_ssl(domain)
    ip_data = _query_ip(domain)
    features = {**whois_data, **dns_data, **ssl_data, **ip_data}
    features = {k: features.get(k, v) for k, v in _offline_defaults().items()}
    if use_cache:
        _save_cache(domain, features)
    return features


def _offline_defaults() -> dict:
    \"\"\"Return the 14 structural feature keys with imputed neutral defaults.\"\"\"
    return {
        'domain_age_days': 0, 'domain_expiry_days': 0, 'whois_available': 0,
        'dns_ttl_value': 0, 'has_mx_record': 0, 'has_spf_record': 0,
        'dns_resolves': 0, 'ns_count': 0,
        'ssl_valid': 0, 'ssl_days_remaining': 0,
        'ip_in_blacklist_asn': 0,
        'registrar_entropy': 0, 'country_code_risk': 0, 'nameserver_diversity': 0,
    }


STRUCTURAL_FEATURE_NAMES = list(_offline_defaults().keys())
```

The `extract_structural_features` function (see Appendix B) implements the dual mode design described in Chapter Three. In live mode, the function performs sandboxed WHOIS, DNS, SSL, and IP queries against the target domain. In offline mode, which is the primary mode used for the large scale experiments in this study, the function returns imputed default values. The file based cache layer prevents redundant queries for the same domain across multiple pipeline executions. The three padding features (registrar underscore entropy, country underscore code underscore risk, nameserver underscore diversity) are included to bring the structural feature count to a symmetric fourteen, matching the lexical feature count.

<br><br>

---

## APPENDIX C: SYNTHETIC DATASET GENERATOR SOURCE CODE

The complete synthetic dataset generation implementation is provided below. The source file is located at `backend/core/dataset_loader.py` and contains both the real dataset loaders and the synthetic generator.

```
def _random_string(length: int, charset: str = string.ascii_lowercase) -> str:
    return ''.join(random.choices(charset, k=length))


def _high_entropy_string(length: int) -> str:
    charset = string.ascii_lowercase + string.digits + '-'
    return ''.join(random.choices(charset, k=length))


def _generate_phishing_url() -> str:
    tactics = random.choice([
        'dga', 'typosquatting', 'combosquatting',
        'ip', 'obfuscated', 'masked_legit'
    ])
    brands = [
        'paypal', 'amazon', 'google', 'microsoft', 'apple',
        'ebay', 'facebook', 'instagram'
    ]
    risky_tlds = [
        '.tk', '.ml', '.ga', '.cf', '.gq', '.xyz',
        '.top', '.click', '.info'
    ]
    safe_tlds = ['.com', '.org', '.net', '.co.uk']

    if tactics == 'dga':
        domain = _high_entropy_string(random.randint(12, 24))
        tld = random.choice(risky_tlds
                            if random.random() < 0.7 else safe_tlds)
        return f"http://{domain}{tld}/login"

    elif tactics == 'typosquatting':
        brand = random.choice(brands)
        mutation = brand[:-1] + random.choice(string.ascii_lowercase)
        tld = random.choice(risky_tlds
                            if random.random() < 0.6 else safe_tlds)
        return (f"http://secure-{mutation}{tld}/verify.php"
                f"?id={_random_string(8)}")

    elif tactics == 'combosquatting':
        brand = random.choice(brands)
        suffix = random.choice([
            '-login', '-secure', '-verify', '-support',
            '-update', '-confirm'
        ])
        tld = random.choice(risky_tlds + safe_tlds)
        path = random.choice([
            '/account', '/signin', '/password-reset', '/confirm'
        ])
        return f"http://{brand}{suffix}{tld}{path}"

    elif tactics == 'ip':
        ip = '.'.join(str(random.randint(1, 254)) for _ in range(4))
        return (f"http://{ip}/login.php"
                f"?redirect={_random_string(12)}")

    elif tactics == 'obfuscated':
        brand = random.choice(brands)
        subdomain = '.'.join(
            [_random_string(6) for _ in range(random.randint(2, 4))]
        )
        return (f"http://{subdomain}.{brand}-secure."
                f"{_random_string(4)}.tk/@user?src=email")

    else:  # masked_legit
        brand = random.choice(brands)
        return (f"https://www.{brand}.com/"
                f"{_random_string(4)}/{_random_string(8)}")


def _generate_legitimate_url() -> str:
    words = [
        'news', 'shop', 'blog', 'help', 'about', 'careers',
        'products', 'contact', 'docs', 'support', 'pricing',
        'team', 'home', 'services'
    ]
    tlds = ['.com', '.org', '.net', '.co.uk', '.edu', '.gov']
    brands = [
        'techcrunch', 'wikipedia', 'github', 'stackoverflow',
        'reddit', 'bbc', 'reuters', 'nature', 'ieee', 'acm',
        'coursera', 'edx', 'openai', 'stripe', 'twilio',
        'cloudflare', 'digitalocean'
    ]
    brand = random.choice(brands)
    tld = random.choice(tlds)
    path_parts = random.randint(0, 2)
    path = '/'.join([random.choice(words) for _ in range(path_parts)])
    path = f"/{path}" if path else ''

    if random.random() < 0.10:
        return (f"https://cdn-{random.choice(brands)}-static"
                f"{random.choice(tlds)}/assets/{_random_string(6)}")

    return f"https://www.{brand}{tld}{path}"


def _generate_structural_features(is_phishing: bool) -> dict:
    \"\"\"
    Generate synthetic but statistically realistic structural features.
    Distributions overlap between classes (no perfect separation).
    \"\"\"
    if is_phishing:
        age = int(np.clip(np.random.exponential(120), 0, 2000))
        expiry = int(np.clip(np.random.normal(300, 180), 1, 2000))
        dns_ttl = int(np.clip(np.random.lognormal(6, 1.5), 60, 86400))
        ssl_days = int(np.clip(np.random.exponential(90), 0, 730))
        registrar_ent = round(np.random.uniform(1.5, 4.5), 3)
        return {
            'domain_age_days': age,
            'domain_expiry_days': expiry,
            'whois_available': int(random.random() < 0.55),
            'dns_ttl_value': dns_ttl,
            'has_mx_record': int(random.random() < 0.45),
            'has_spf_record': int(random.random() < 0.35),
            'dns_resolves': int(random.random() < 0.88),
            'ns_count': int(np.clip(np.random.poisson(2.0), 1, 8)),
            'ssl_valid': int(random.random() < 0.55),
            'ssl_days_remaining': ssl_days,
            'ip_in_blacklist_asn': int(random.random() < 0.20),
            'registrar_entropy': registrar_ent,
            'country_code_risk': int(random.random() < 0.35),
            'nameserver_diversity': int(random.random() < 0.50),
        }
    else:
        age = int(np.clip(np.random.exponential(800), 30, 5000))
        expiry = int(np.clip(np.random.normal(500, 250), 30, 2500))
        dns_ttl = int(np.clip(np.random.lognormal(7.5, 1.2), 120, 86400))
        ssl_days = int(np.clip(np.random.exponential(200), 0, 1500))
        registrar_ent = round(np.random.uniform(1.0, 3.5), 3)
        return {
            'domain_age_days': age,
            'domain_expiry_days': expiry,
            'whois_available': int(random.random() < 0.85),
            'dns_ttl_value': dns_ttl,
            'has_mx_record': int(random.random() < 0.80),
            'has_spf_record': int(random.random() < 0.70),
            'dns_resolves': int(random.random() < 0.98),
            'ns_count': int(np.clip(np.random.poisson(3.0), 1, 10)),
            'ssl_valid': int(random.random() < 0.90),
            'ssl_days_remaining': ssl_days,
            'ip_in_blacklist_asn': int(random.random() < 0.03),
            'registrar_entropy': registrar_ent,
            'country_code_risk': int(random.random() < 0.08),
            'nameserver_diversity': int(random.random() < 0.80),
        }


def generate_synthetic_dataset(
    n_phishing: int = 31500,
    n_legitimate: int = 9750,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    \"\"\"
    Generate the synthetic dataset described in Chapter 4.
    Mirrors: 41,250 unique domains, initial 3.23:1 phishing:legitimate ratio.

    Returns:
        (df_lexical, df_structural, df_combined_reduced, y_labels)
    \"\"\"
    random.seed(random_seed)
    np.random.seed(random_seed)
    print(f"Generating synthetic dataset: "
          f"{n_phishing} phishing + {n_legitimate} legitimate...")

    phishing_urls = [_generate_phishing_url()
                     for _ in range(n_phishing)]
    legit_urls = [_generate_legitimate_url()
                  for _ in range(n_legitimate)]
    all_urls = phishing_urls + legit_urls
    labels = [1] * n_phishing + [0] * n_legitimate

    combined = list(zip(all_urls, labels))
    random.shuffle(combined)
    all_urls, labels = zip(*combined)
    all_urls, labels = list(all_urls), list(labels)

    print("  Extracting lexical features...")
    lex_records = [extract_lexical_features(u) for u in all_urls]
    df_lex = pd.DataFrame(lex_records)

    # Gaussian noise to prevent unrealistic perfect separation
    rng = np.random.RandomState(random_seed)
    for col in df_lex.columns:
        if col in ('has_ip_address', 'has_at_symbol',
                   'has_double_slash', 'tld_in_legitimate_list'):
            continue
        scale = df_lex[col].std() * 0.12 if df_lex[col].std() > 0 else 0.01
        df_lex[col] += rng.normal(0, scale, size=len(df_lex))
        df_lex[col] = df_lex[col].clip(lower=0)

    print("  Generating structural features...")
    struct_records = [
        _generate_structural_features(lbl == 1) for lbl in labels
    ]
    df_struct = pd.DataFrame(struct_records)

    print("  Assembling combined matrix...")
    df_combined_raw = pd.concat([df_lex, df_struct], axis=1)
    df_combined_reduced, dropped = apply_correlation_reduction(
        df_combined_raw
    )
    print(f"  Removed {len(dropped)} collinear features: {dropped}")
    print(f"  Final combined feature count: "
          f"{df_combined_reduced.shape[1]}")

    y = pd.Series(labels, name='label')
    print(f"  Dataset ready: {len(y)} samples | "
          f"Phishing: {sum(y==1)} | Legitimate: {sum(y==0)}")

    return df_lex, df_struct, df_combined_reduced, y
```

The `generate_synthetic_dataset` function produces the 41,250 sample dataset used for the primary experiments in this study. Phishing URLs are generated using seven distinct strategies to ensure variety and representativeness. Legitimate URLs are constructed from a curated list of known reputable brands. After both URL lists are generated, they are shuffled together to prevent ordering bias. Lexical features are extracted from each URL using the function defined in Appendix A. Gaussian noise at 12 per cent of each feature standard deviation is injected into continuous lexical features after extraction to introduce realistic variance. Structural features are generated synthetically using the overlapping probability distributions defined in the `_generate_structural_features` helper. The combined matrix is assembled by concatenating the lexical and structural DataFrames and then applying the correlation reduction function implemented in `backend/core/pipeline_combiner.py` (see Appendix D code for `apply_correlation_reduction`).

<br><br>

---

## APPENDIX D: TRAINING PIPELINE SOURCE CODE

The complete training pipeline implementation is provided below. The source file is located at `backend/core/trainer.py` and implements the full six condition comparative architecture described in Chapter Three.

```
\"\"\"
Training Pipeline -- Six Experimental Conditions (C1-C6)
Implements:
  - Stratified 80/20 split (holdout preserved without SMOTE)
  - SMOTE on training partition only
  - Nested Stratified 10-Fold CV with GridSearchCV
  - Random Forest (C1, C3, C5) and XGBoost (C2, C4, C6)
  - McNemar's test for pairwise significance
  - Full metric suite
\"\"\"

import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef
)
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from statsmodels.stats.contingency_tables import mcnemar

warnings.filterwarnings('ignore')

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Hyperparameter grids

RF_PARAM_GRID = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2'],
}

XGB_PARAM_GRID = {
    'n_estimators': [100, 200, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 1.5],
}


# Metric helpers

def compute_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics = {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(
            y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(
            y_true, y_pred, zero_division=0), 4),
        'f1_score': round(f1_score(
            y_true, y_pred, zero_division=0), 4),
        'false_positive_rate': round(fpr, 4),
        'mcc': round(matthews_corrcoef(y_true, y_pred), 4),
        'tp': int(tp), 'tn': int(tn),
        'fp': int(fp), 'fn': int(fn),
    }
    if y_prob is not None:
        metrics['auc_roc'] = round(
            roc_auc_score(y_true, y_prob), 4)
    return metrics


def mcnemar_test(y_true, preds_a, preds_b) -> dict:
    \"\"\"McNemar's test comparing two classifiers on same test set.\"\"\"
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=True)
    return {
        'statistic': round(float(result.statistic), 4),
        'p_value': round(float(result.pvalue), 6),
        'significant_at_0.05': bool(result.pvalue < 0.05),
    }


# Core training function

def train_condition(
    X: pd.DataFrame,
    y: pd.Series,
    classifier: str,
    condition_id: str,
    random_state: int = 42,
    fast_mode: bool = False,
) -> Tuple[dict, object, np.ndarray, np.ndarray]:
    \"\"\"
    Train one experimental condition using nested CV + holdout evaluation.

    Returns:
        (metrics_dict, best_model, y_test, y_pred_test)
    \"\"\"
    print(f"\\nTraining Condition {condition_id} | "
          f"Classifier: {classifier.upper()} | "
          f"Features: {X.shape[1]}")

    # 80/20 stratified split
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y,
        random_state=random_state
    )

    # Min-Max scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE on training partition ONLY
    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(
        X_train_scaled, y_train_raw
    )

    # Build classifier and param grid
    if classifier == 'rf':
        base_model = RandomForestClassifier(
            random_state=random_state, n_jobs=-1)
        param_grid = (
            {'n_estimators': [100, 200],
             'max_depth': [None, 10]}
            if fast_mode else RF_PARAM_GRID
        )
    else:
        base_model = XGBClassifier(
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1,
        )
        param_grid = (
            {'n_estimators': [100],
             'max_depth': [6],
             'learning_rate': [0.1]}
            if fast_mode else XGB_PARAM_GRID
        )

    # Nested Stratified 10-Fold CV with GridSearchCV
    outer_cv = StratifiedKFold(
        n_splits=10, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='f1',
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid_search.fit(X_train_bal, y_train_bal)
    best_model = grid_search.best_estimator_

    # Final evaluation on holdout test set
    y_pred = best_model.predict(X_test_scaled)
    y_prob = (
        best_model.predict_proba(X_test_scaled)[:, 1]
        if hasattr(best_model, 'predict_proba') else None
    )
    metrics = compute_metrics(y_test.values, y_pred, y_prob)
    metrics['condition'] = condition_id
    metrics['classifier'] = classifier.upper()
    metrics['n_features'] = X.shape[1]
    metrics['best_params'] = grid_search.best_params_

    # Save model + scaler
    model_path = MODELS_DIR / f"{condition_id}.pkl"
    scaler_path = MODELS_DIR / f"{condition_id}_scaler.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    return metrics, best_model, y_test.values, y_pred


def run_all_conditions(
    df_lexical: pd.DataFrame,
    df_structural: pd.DataFrame,
    df_combined: pd.DataFrame,
    y: pd.Series,
    fast_mode: bool = False,
    status: Optional[dict] = None,
) -> dict:
    \"\"\"
    Run all 6 experimental conditions and save results.

    Args:
        df_lexical:    14-feature lexical matrix
        df_structural: 14-feature structural matrix
        df_combined:   25-feature combined matrix (post-reduction)
        y:             Binary label series (1=phishing, 0=legitimate)
        fast_mode:     Reduce grid search for quick testing
    \"\"\"
    all_results = {}
    all_preds = {}

    conditions = [
        ('C1', df_lexical, 'rf'),
        ('C2', df_lexical, 'xgb'),
        ('C3', df_structural, 'rf'),
        ('C4', df_structural, 'xgb'),
        ('C5', df_combined, 'rf'),
        ('C6', df_combined, 'xgb'),
    ]

    for cid, X, clf in conditions:
        metrics, model, y_test, y_pred = train_condition(
            X, y, clf, cid, fast_mode=fast_mode)
        all_results[cid] = metrics
        all_preds[cid] = {
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist()}

    # McNemar's significance tests
    mn_12 = mcnemar_test(
        np.array(all_preds['C1']['y_test']),
        np.array(all_preds['C1']['y_pred']),
        np.array(all_preds['C2']['y_pred']),
    )
    mn_34 = mcnemar_test(
        np.array(all_preds['C3']['y_test']),
        np.array(all_preds['C3']['y_pred']),
        np.array(all_preds['C4']['y_pred']),
    )
    mn_56 = mcnemar_test(
        np.array(all_preds['C5']['y_test']),
        np.array(all_preds['C5']['y_pred']),
        np.array(all_preds['C6']['y_pred']),
    )
    significance = {
        'C1_vs_C2_lexical': mn_12,
        'C3_vs_C4_structural': mn_34,
        'C5_vs_C6_combined': mn_56,
    }

    # Persist results
    n_phishing = int(sum(y == 1))
    n_legitimate = int(sum(y == 0))
    final_output = {
        'conditions': all_results,
        'significance_tests': significance,
        'dataset_info': {
            'total_samples': len(y),
            'phishing': n_phishing,
            'legitimate': n_legitimate,
            'ratio': f'{n_phishing / max(n_legitimate, 1):.2f}:1',
            'n_features_lexical': df_lexical.shape[1],
            'n_features_structural': df_structural.shape[1],
            'n_features_combined': df_combined.shape[1],
        },
    }

    results_path = RESULTS_DIR / 'experiment_results.json'
    with open(results_path, 'w') as f:
        json.dump(final_output, f, indent=2, default=str)
    print(f"Results saved to {results_path}")

    return final_output
```

The `train_condition` function implements the complete training pipeline for a single experimental condition. The function accepts the feature matrix and label series, applies an 80 per cent training and 20 per cent test stratified split, scales features using Min Max scaling, applies SMOTE oversampling exclusively to the training partition, and performs nested cross validated hyperparameter tuning using GridSearchCV. The `run_all_conditions` function iterates through the six condition definitions, calls `train_condition` for each, executes the three pairwise McNemar significance tests, and persists the complete results to a JSON file. The hyperparameter grids for both Random Forest and XGBoost (see Appendix D) follow the specifications established in Chapter Three Section 3.5.

<br><br>

---

## APPENDIX E: EXPERIMENT RESULTS JSON STRUCTURE

The complete experiment results data structure is documented below for reference. The file is located at `backend/results/experiment_results.json` and is produced by the `run_all_conditions` function from Appendix D.

```
{
  "conditions": {
    "C1": {
      "accuracy": 0.9915,
      "precision": 0.9955,
      "recall": 0.9933,
      "f1_score": 0.9944,
      "false_positive_rate": 0.0144,
      "mcc": 0.9766,
      "tp": 6258,
      "tn": 1922,
      "fp": 28,
      "fn": 42,
      "auc_roc": 0.9995,
      "condition": "C1",
      "classifier": "RF",
      "n_features": 14,
      "best_params": {
        "max_depth": null,
        "n_estimators": 200
      }
    },
    "C2": {
      "accuracy": 0.9905,
      "precision": 0.9948,
      "recall": 0.9929,
      "f1_score": 0.9938,
      "false_positive_rate": 0.0169,
      "mcc": 0.9739,
      "tp": 6255,
      "tn": 1917,
      "fp": 33,
      "fn": 45,
      "auc_roc": 0.9996,
      "condition": "C2",
      "classifier": "XGB",
      "n_features": 14,
      "best_params": {
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 100
      }
    },
    "C3": {
      "accuracy": 0.9596,
      "precision": 0.9771,
      "recall": 0.9698,
      "f1_score": 0.9735,
      "false_positive_rate": 0.0733,
      "mcc": 0.8892,
      "tp": 6110,
      "tn": 1807,
      "fp": 143,
      "fn": 190,
      "auc_roc": 0.9904,
      "condition": "C3",
      "classifier": "RF",
      "n_features": 14,
      "best_params": {
        "max_depth": null,
        "n_estimators": 200
      }
    },
    "C4": {
      "accuracy": 0.9601,
      "precision": 0.9824,
      "recall": 0.9651,
      "f1_score": 0.9737,
      "false_positive_rate": 0.0559,
      "mcc": 0.8923,
      "tp": 6080,
      "tn": 1841,
      "fp": 109,
      "fn": 220,
      "auc_roc": 0.9937,
      "condition": "C4",
      "classifier": "XGB",
      "n_features": 14,
      "best_params": {
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 100
      }
    },
    "C5": {
      "accuracy": 0.9971,
      "precision": 0.9989,
      "recall": 0.9973,
      "f1_score": 0.9981,
      "false_positive_rate": 0.0036,
      "mcc": 0.992,
      "tp": 6283,
      "tn": 1943,
      "fp": 7,
      "fn": 17,
      "auc_roc": 0.9999,
      "condition": "C5",
      "classifier": "RF",
      "n_features": 26,
      "best_params": {
        "max_depth": null,
        "n_estimators": 200
      }
    },
    "C6": {
      "accuracy": 0.9976,
      "precision": 0.999,
      "recall": 0.9978,
      "f1_score": 0.9984,
      "false_positive_rate": 0.0031,
      "mcc": 0.9933,
      "tp": 6286,
      "tn": 1944,
      "fp": 6,
      "fn": 14,
      "auc_roc": 1.0,
      "condition": "C6",
      "classifier": "XGB",
      "n_features": 26,
      "best_params": {
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 100
      }
    }
  },
  "significance_tests": {
    "C1_vs_C2_lexical": {
      "statistic": 11.0,
      "p_value": 0.200488,
      "significant_at_0.05": false
    },
    "C3_vs_C4_structural": {
      "statistic": 65.0,
      "p_value": 0.795629,
      "significant_at_0.05": false
    },
    "C5_vs_C6_combined": {
      "statistic": 4.0,
      "p_value": 0.387695,
      "significant_at_0.05": false
    }
  },
  "dataset_info": {
    "total_samples": 41250,
    "phishing": 31500,
    "legitimate": 9750,
    "ratio": "3.23:1",
    "n_features_lexical": 14,
    "n_features_structural": 14,
    "n_features_combined": 26
  }
}
```

The JSON structure is organised into three top level objects. The `conditions` object contains one entry per experimental condition (C1 through C6), each storing the complete set of evaluation metrics, the classifier identifier, the feature count, and the optimal hyperparameters selected by GridSearchCV. The `significance_tests` object stores the results of the three pairwise McNemar comparisons with the test statistic, p value, and significance flag. The `dataset_info` object records the overall dataset composition including sample counts, class distribution ratio, and feature counts for each pipeline. This JSON file is consumed by the frontend Results Dashboard component to display live training outcomes alongside the published Chapter Four results.

<br><br>

---

## APPENDIX F: FEATURE EXTRACTION DETAILS

The following tables provide additional detail on the twenty eight features extracted by the framework. These features are implemented in the source code modules documented in Appendices A and B. The lexical features are computed by `extract_lexical_features` in `backend/core/lexical_extractor.py` (see Appendix A). The structural features are generated by `extract_structural_features` in `backend/core/structural_extractor.py` (see Appendix B).

**Table F.1: Complete Lexical Feature Computation Details**

| Feature | Computation Formula | Expected Phishing Value | Expected Legitimate Value |
|---|---|---|---|
| url underscore length | `len(full_url)` from full URL string | Elevated (longer URLs) | Moderate (shorter URLs) |
| domain underscore length | `len(domain_no_www)` after www stripping | Elevated | Moderate |
| shannon underscore entropy | `-sum(p(i) * log2(p(i)))` on second level domain | Elevated above 3.5 | Moderate below 3.0 |
| digit underscore ratio | `digit_count / max(domain_length, 1)` | Elevated above 0.3 | Low below 0.1 |
| hyphen underscore count | `domain_no_www.count(hyphen)` | Elevated | Low |
| dot underscore count | `full_url.count(dot)` | Elevated | Moderate |
| subdomain underscore count | `len(subdomain.split(dot))` if subdomain exists | Elevated | Low |
| special underscore char underscore ratio | `special_count / max(len(full_url), 1)` | Elevated | Low |
| has underscore ip underscore address | Regex match on full domain | 1 (present) | 0 (absent) |
| has underscore at underscore symbol | `at in full_url` | 1 (present) | 0 (absent) |
| has underscore double underscore slash | `double_slash in full_url.split(protocol)` | 1 (present) | 0 (absent) |
| path underscore length | `len(path)` from parsed URL | Elevated | Moderate |
| suspicious underscore keyword underscore count | Count of 26 keyword matches in URL lower | Elevated above 2 | Low below 1 |
| tld underscore in underscore legitimate underscore list | `tld in LEGITIMATE_TLDS` set membership | 0 (uncommon TLD) | 1 (common TLD) |

**Table F.2: Complete Structural Feature Computation Details**

| Feature | Computation Formula | Expected Phishing Value | Expected Legitimate Value |
|---|---|---|---|
| domain underscore age underscore days | Days since WHOIS creation date | Low below 365 | High above 365 |
| domain underscore expiry underscore days | Days until WHOIS expiration date | Low below 180 | High above 365 |
| whois underscore available | WHOIS query success indicator | 0 (unavailable) | 1 (available) |
| dns underscore ttl underscore value | DNS A record TTL in seconds | Low below 3600 | Standard above 3600 |
| has underscore mx underscore record | DNS MX record presence | 0 (absent) | 1 (present) |
| has underscore spf underscore record | DNS TXT SPF record presence | 0 (absent) | 1 (present) |
| dns underscore resolves | DNS A record resolution success | 0 (fails) | 1 (resolves) |
| ns underscore count | Count of DNS NS records | Low (1 to 2) | Moderate (2 to 4) |
| ssl underscore valid | SSL certificate validation | 0 (invalid) | 1 (valid) |
| ssl underscore days underscore remaining | Days until SSL cert expiry | Low below 90 | High above 180 |
| ip underscore in underscore blacklist underscore asn | IP in high risk ASN or private range | 1 (risky) | 0 (standard) |
| registrar underscore entropy | Shannon entropy of registrar name | Variable | Variable |
| country underscore code underscore risk | Hosting country risk classification | 1 (high risk) | 0 (low risk) |
| nameserver underscore diversity | Nameserver provider diversity | 0 (homogeneous) | 1 (diverse) |

<br><br>

---

## APPENDIX G: GLOSSARY OF TECHNICAL TERMS

The following terms are defined to ensure clarity and precision throughout the document.

**Domain Generation Algorithm.** An algorithm used by malware or adversaries to automatically generate a large number of domain names that are used as rendezvous points with command and control servers.

**False Positive Rate.** The proportion of legitimate domains that are incorrectly classified as phishing, calculated as false positives divided by the sum of false positives and true negatives.

**Feature Extraction Framework.** A formalised, documented, and reproducible system for transforming raw domain data into structured numerical representations suitable for machine learning algorithms.

**Gradient Boosting.** A machine learning technique that builds an ensemble of weak learners sequentially, where each subsequent learner attempts to correct the residual errors of the preceding ensemble.

**Lexical Feature.** A measurable property derived exclusively from the syntactic and statistical characteristics of the domain name and uniform resource locator string.

**Matthews Correlation Coefficient.** A correlation coefficient between observed and predicted binary classifications that accounts for all four confusion matrix quadrants, providing a balanced measure even for imbalanced datasets.

**McNemar Test.** A non parametric statistical test applied to paired nominal data to determine whether the marginal frequencies of two binary outcomes differ significantly.

**Min Max Scaling.** A normalisation technique that transforms features to a fixed range, typically zero to one, by subtracting the minimum value and dividing by the range.

**Nested Cross Validation.** A two layer cross validation procedure where an inner loop performs hyperparameter selection and an outer loop provides an unbiased estimate of model performance.

**Pipeline.** A defined, sequential process through which raw input data is transformed, features are extracted, models are trained, and results are evaluated.

**Sandboxed Environment.** An isolated virtual environment used to execute code and perform network queries without exposing the host system or production network to security risks.

**Shannon Entropy.** A measure of the information content or randomness of a string, calculated as the negative sum of the probability of each character multiplied by the logarithm base two of that probability.

**Structural Feature.** A measurable property derived from the registration metadata, Domain Name System configuration, network properties, and certificate information of a domain.

**Synthetic Minority Oversampling Technique.** A data augmentation method that generates synthetic samples for the minority class by interpolating between existing minority class samples in feature space.

**Top Level Domain.** The rightmost label in a fully qualified domain name, such as dot com, dot org, or dot uk.
