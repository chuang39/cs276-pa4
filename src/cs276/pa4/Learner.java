package cs276.pa4;

import static cs276.pa4.Util.loadRelData;
import static cs276.pa4.Util.loadTrainData;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public abstract class Learner {
	
	// Smoothing factor for log transformation to pageranks
	private static final double LAMBDA = 3.0;
	
	/* Construct training features matrix */
	public abstract Instances extract_train_features(String train_data_file, String train_rel_file, Map<String,Double> idfs);

	/* Train the model */
	public abstract Classifier training (Instances dataset);
	
	/* Construct testing features matrix */
	public abstract TestFeatures extract_test_features(String test_data_file, Map<String,Double> idfs);
	
	/* Test the model, return ranked queries */
	public abstract Map<String, List<String>> testing(TestFeatures tf, Classifier model);
	
	/*
	 * A generic feature extract function for both train and test.
	 */
	TestFeatures extractFeatures(String datasetName, String dataFile, String relFile, Map<String, Double> idfs) {
		
		Instances features = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		features = new Instances(datasetName, attributes, 0);
		
		Map<Query, List<Document>> data = null;
		Map<String, Map<String, Double>> rels = null;
		try {
			data = loadTrainData(dataFile);
			if (relFile != null)
				rels = loadRelData(relFile);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		Map<String, Map<String, Integer>> indexMap = new HashMap<String, Map<String, Integer>>();
		int index = 0;
		for (Query q : data.keySet()) {
			List<Document> docs = data.get(q);
			index = extractDocumentFeatures(features, q, docs, idfs, rels, indexMap, index);
		}
		
		/* Set last attribute as target */
		features.setClassIndex(features.numAttributes() - 1);
		
		TestFeatures testFeatures = new TestFeatures();
		testFeatures.features = features;
		testFeatures.index_map = indexMap;		
		
		return testFeatures;
	}

	int extractDocumentFeatures(Instances features, Query q, List<Document> docs,
			Map<String, Double> idfs, Map<String, Map<String, Double>> rels,
			Map<String, Map<String, Integer>> indexMap, int index) {
		Map<String, Integer> indices = new HashMap<>();
		String query = q.toString();

		for (Document doc : docs) {
			String url = doc.url;
			double relScore = rels == null ? 0.0 : rels.get(query).get(url);

			List<Double> featureVector = extractTfIdfFeatures(q, doc, idfs);

			// Form final feature vector by adding relevance score
			featureVector.add(relScore);

			// Copy list into an array of doubles
			double[] featureVecArr = new double[featureVector.size()];

			for (int i = 0; i < featureVector.size(); i++) {
				featureVecArr[i] = featureVector.get(i);
			}

			Instance inst = new DenseInstance(1.0, featureVecArr);
			features.add(inst);
			indices.put(url, index++);
		}

		indexMap.put(query, indices);

		return index;
	}
	
	/**
	 * Returns a list of the tfidf cosine similarity features for each field
	 * 
	 * @param q
	 * @param doc
	 * @param idfs
	 * @return
	 */
	static List<Double> extractTfIdfFeatures(Query q, Document doc, Map<String, Double> idfs) {
		// TODO: we should use BM25 to calculate tf!
		Map<String, Map<String, Double>> tfs = getDocTermFreqs(doc, q);
		normalizeTFs(tfs, doc, q);

		Map<String, Double> tfQuery = getQueryFreqs(q, idfs);
		
		List<Double> tfidfFeatures = new ArrayList<Double>();
        
    tfidfFeatures.add(dotProduct(tfQuery, tfs.get("url")));
    tfidfFeatures.add(dotProduct(tfQuery, tfs.get("title")));
    tfidfFeatures.add(dotProduct(tfQuery, tfs.get("body")));
    tfidfFeatures.add(dotProduct(tfQuery, tfs.get("header")));
    tfidfFeatures.add(dotProduct(tfQuery, tfs.get("anchor")));

    return tfidfFeatures;
	}
	
	static double dotProduct(Map<String, Double> tsv, Map<String, Double> qv) {
		/*
		 * iterate the terms to get the result of dot product
		 */
		double sum = 0.0;
		for (Map.Entry<String, Double> entry : tsv.entrySet()) {
			String key = entry.getKey();
			Double docTermVector = entry.getValue();
			Double queryTermVector = qv.containsKey(key) ? qv.get(key) : 0.0;
			sum += docTermVector * queryTermVector;
		}
		return sum;
	}
	
	
	// Handle the query vector to get query frequency
	public static Map<String, Double> getQueryFreqs(Query q, Map<String, Double> idfs) {
		Map<String, Double> tfQuery = new HashMap<String, Double>(); // queryWord -> term frequency
		Map<String, Integer> counts = new HashMap<String, Integer>();

		List<String> queryWords = q.words;

		for (String queryWord : queryWords) {
			if (counts.containsKey(queryWord)) {
				// Increment the term doc count
				counts.put(queryWord, counts.get(queryWord) + 1);
			} else {
				counts.put(queryWord, 1);
			}
		}

        for (String term : counts.keySet()) {
            if (idfs.containsKey(term)) {
                tfQuery.put(term, 1.0 * counts.get(term) * idfs.get(term));
            } else {
                tfQuery.put(term, Math.log(98998+1));
            }
        }
		return tfQuery;
	}
	
	public static void normalizeTFs(Map<String,Map<String, Double>> tfs, Document d, Query q) {
		
		// Normalize document tf vectors by length with smoothing
		for (String tfType : tfs.keySet()) {
			
			Map<String, Double> termToFreq = tfs.get(tfType);
			
			for (String term : termToFreq.keySet()) {
				termToFreq.put(term, termToFreq.get(term) / (double)(d.body_length + 500));
			}
		}
	}
	
	static String[] TFTYPES = { "url", "title", "body", "header", "anchor" };

	public static Map<String, Map<String, Double>> getDocTermFreqs(Document d, Query q) {
		// Map from tf type -> queryWord -> score
		Map<String, Map<String, Double>> tfs = new HashMap<String, Map<String, Double>>();

		// //////////////////Initialization/////////////////////

		// Initialize map from string to count for URL
		Map<String, Double> urlTfs = new HashMap<String, Double>();

		if (d.url != null) {
			List<String> urlTokens = Arrays.asList(d.url.toLowerCase().split("\\W+"));

			for (String urlToken : urlTokens) {
				if (urlTfs.containsKey(urlToken)) {
					// Increment the term doc count
					urlTfs.put(urlToken, urlTfs.get(urlToken) + 1);
				} else {
					urlTfs.put(urlToken, 1.0);
				}
			}
		}

		// Initialize map from string to count for title
		Map<String, Double> titleTfs = new HashMap<String, Double>();

		if (d.title != null) {
			List<String> titleTokens = Arrays.asList(d.title.toLowerCase().split(
					"\\s+"));

			for (String titleToken : titleTokens) {
				if (titleTfs.containsKey(titleToken)) {
					// Increment the term doc count
					titleTfs.put(titleToken, titleTfs.get(titleToken) + 1);
				} else {
					titleTfs.put(titleToken, 1.0);
				}
			}
		}

		// Initialize map from string to count for body
		Map<String, Double> bodyTfs = new HashMap<String, Double>();

		if (d.body_hits != null) {
			Map<String, List<Integer>> bodyHits = d.body_hits;

			for (String bodyHit : bodyHits.keySet()) {
				bodyTfs.put(bodyHit, (double) bodyHits.get(bodyHit).size());
			}
		}

		// Initialize map from string to count for header
		Map<String, Double> headerTfs = new HashMap<String, Double>();

		if (d.headers != null) {
			List<String> headers = d.headers;

			for (String header : headers) {
				List<String> headerTokens = Arrays.asList(header.toLowerCase().split(
						"\\s+"));

				for (String headerToken : headerTokens) {
					if (headerTfs.containsKey(headerToken)) {
						// Increment the term doc count
						headerTfs.put(headerToken, headerTfs.get(headerToken) + 1);
					} else {
						headerTfs.put(headerToken, 1.0);
					}
				}

			}
		}

		// Initialize map from string to count for anchor
		Map<String, Double> anchorTfs = new HashMap<String, Double>();

		if (d.anchors != null) {
			Map<String, Integer> anchors = d.anchors;

			for (String anchor : anchors.keySet()) {

				List<String> anchorTokens = Arrays.asList(anchor.toLowerCase().split(
						"\\s+"));

				int anchorCount = anchors.get(anchor);

				for (String anchorToken : anchorTokens) {
					if (anchorTfs.containsKey(anchorToken)) {
						// Increment the term doc count
						anchorTfs.put(anchorToken, anchorTfs.get(anchorToken) + anchorCount);
					} else {
						anchorTfs.put(anchorToken, (double) anchorCount);
					}
				}

			}
		}
		
		// Add an empty map for each type
		for (String tfType : TFTYPES) {
			HashMap<String, Double> stringToTf = new HashMap<String, Double>();
			tfs.put(tfType, stringToTf);
		}

		// //////////////////////////////////////////////////////

		// Loop through query terms and increase relevant tfs. Note: you should do
		// this to each type of term frequencies.
		for (String queryWord : q.words) {

			// URL
			if (urlTfs.containsKey(queryWord)) {
				tfs.get(TFTYPES[0]).put(queryWord, urlTfs.get(queryWord));
			}

			// Title
			if (titleTfs.containsKey(queryWord)) {
				tfs.get(TFTYPES[1]).put(queryWord, titleTfs.get(queryWord));
			}

			// Body
			if (bodyTfs.containsKey(queryWord)) {
				tfs.get(TFTYPES[2]).put(queryWord, bodyTfs.get(queryWord));
			}

			// Header
			if (headerTfs.containsKey(queryWord)) {
				tfs.get(TFTYPES[3]).put(queryWord, headerTfs.get(queryWord));
			}

			// Anchor
			if (anchorTfs.containsKey(queryWord)) {
				tfs.get(TFTYPES[4]).put(queryWord, anchorTfs.get(queryWord));
			}

		}
		return tfs;
	}
	
	// Added new versions of existing methods to handle all features.
	

	/**
	 * Extracts all the features from the data file and returns them. This includes tfidf cosine
	 * similarity features as well as any additional features.
	 * 
	 * @param datasetName
	 * @param dataFile
	 * @param relFile
	 * @param idfs
	 * @return
	 */
	TestFeatures extractAllFeatures(String datasetName, String dataFile, String relFile, Map<String, Double> idfs) {
		
		Instances features = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		
		// New features
		attributes.add(new Attribute("bm25_score"));
		attributes.add(new Attribute("pagerank"));
		
		/*
		 * TODO add additional features here
		 * 
		 * Note: The attributes here must match up with the Instances data set returned from this function
		 */
		
		attributes.add(new Attribute("relevance_score"));
		features = new Instances(datasetName, attributes, 0);
		
		Map<Query, List<Document>> data = null;
		Map<String, Map<String, Double>> rels = null;
		try {
			data = loadTrainData(dataFile);
			if (relFile != null)
				rels = loadRelData(relFile);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		Map<String, Map<String, Integer>> indexMap = new HashMap<String, Map<String, Integer>>();
		int index = 0;
		for (Query q : data.keySet()) {
			List<Document> docs = data.get(q);

			index = extractAllDocumentFeatures(features, q, docs, idfs, rels, indexMap, index);
		}
		
		/* Set last attribute as target */
		features.setClassIndex(features.numAttributes() - 1);
		
		TestFeatures testFeatures = new TestFeatures();
		testFeatures.features = features;
		testFeatures.index_map = indexMap;		
		
		return testFeatures;
	}
	
	/**
	 * Extracts all the document features and adds them to features
	 * 
	 * 
	 * @param features Instances that the features will be added to
	 * @param q	The query
	 * @param docs List of documents feature vectors will be built for
	 * @param idfs Idfs
	 * @param rels 
	 * @param indexMap Index map to be updated
	 * @param index Starting index
	 * @return
	 */
	int extractAllDocumentFeatures(Instances features, Query q, List<Document> docs,
			Map<String, Double> idfs, Map<String, Map<String, Double>> rels,
			Map<String, Map<String, Integer>> indexMap, int index) {
		Map<String, Integer> indices = new HashMap<>();
		String query = q.toString();

		for (Document doc : docs) {
			String url = doc.url;
			double relScore = rels == null ? 0.0 : rels.get(query).get(url);
			
			// Extract tfidf features
			List<Double> featureVector = extractTfIdfFeatures(q, doc, idfs);
			
			// Extract additional features
			featureVector.addAll(extractAdditionalFeatures(q, doc, docs, idfs));

			// Form final feature vector by adding relevance score
			featureVector.add(relScore);

			// Copy list into an array of doubles
			double[] featureVecArr = new double[featureVector.size()];

			for (int i = 0; i < featureVector.size(); i++) {
				featureVecArr[i] = featureVector.get(i);
			}

			Instance inst = new DenseInstance(1.0, featureVecArr);
			features.add(inst);
			indices.put(url, index++);
		}

		indexMap.put(query, indices);

		return index;
	}
	
	/**
	 * Returns a list of doubles corresponding to the additional features
	 * 
	 * @param q
	 * @param doc
	 * @return
	 */
	List<Double> extractAdditionalFeatures(Query q, Document doc,
			List<Document> docs, Map<String, Double> idfs) {

		List<Double> features = new ArrayList<Double>();
		
		BM25Scorer bm25Scorer = new BM25Scorer(idfs, docs);
		features.add(bm25Scorer.getSimScore(doc, q));

		// Pageranks must be smoothed because some documents dont have one
		features.add(Math.log(doc.page_rank + LAMBDA));

		/*
		 * TODO add additional features here
		 */

		return features;

	}
}
