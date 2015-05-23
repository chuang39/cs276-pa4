package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cs276.utils.Pair;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import static cs276.pa4.Util.loadTrainData;
import static cs276.pa4.Util.loadRelData;
import static cs276.utils.MapUtils.convertToPairs;

public class PointwiseLearner extends Learner {

	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {
		TestFeatures testFeatures = extractFeatures("train_dataset", train_data_file, train_rel_file, idfs);
        return testFeatures.features;
	}

	@Override
	public Classifier training(Instances dataset) {
		Classifier lr = new LinearRegression();
		try {
			lr.buildClassifier(dataset);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return lr;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) {
		return extractFeatures("test_dataset", test_data_file, null, idfs);
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf, Classifier model) {
		Map<String, List<String>> results = new HashMap<String, List<String>>();

		for (String query : tf.index_map.keySet()) {
			Map<String, Integer> candidates = tf.index_map.get(query);
			List<String> rankings = rankDocuments(query, candidates, model, tf.features);
			results.put(query, rankings);
		}
		return results;
	}

	List<String> rankDocuments(String query, Map<String, Integer> candidates, final Classifier model, final Instances features) {
		// query: url->feature index
		List<Pair<String, Integer>> pairs = convertToPairs(candidates);
		
		Collections.sort(pairs, Collections.reverseOrder(new Comparator<Pair<String,Integer>>(){
			@Override
			public int compare(Pair<String, Integer> o1, Pair<String, Integer> o2) {
				Instance inst1 = features.get(o1.getSecond());
				Instance inst2 = features.get(o2.getSecond());
			
				return compareInstances(inst1, inst2, model);
			}	
		}));
		
		List<String> res = new ArrayList<String>();
		for (Pair<String, Integer> p : pairs) {
			res.add(p.getFirst());
		}
		return res;
	}

	int compareInstances(Instance inst1, Instance inst2, Classifier model) {
		Double score1 = getScore(model, inst1);
	    Double score2 = getScore(model, inst2);

	    // rank larger values higher
	    return score1.compareTo(score2);
	}

	static double getScore(Classifier model, Instance inst) {
        double score = 0;
        try {
            score = model.classifyInstance(inst);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return score;
    }
    
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

	int extractDocumentFeatures(Instances features, Query q, List<Document> docs, Map<String, Double> idfs, Map<String, Map<String, Double>> rels,
								Map<String, Map<String, Integer>> indexMap, int index) {
		Map<String, Integer> indices = new HashMap<>();
	    String query = q.toString();

	    for (Document doc : docs) {
	    	String url = doc.url;
            double relScore = rels == null ? 0.0 : rels.get(query).get(url);
            Instance inst = new DenseInstance(1.0, extractTfIdfFeatures(q, doc, relScore, idfs));
            features.add(inst);
	        indices.put(url, index++);
	    }

        indexMap.put(query, indices);

        return index;
	}
	
	static double[] extractTfIdfFeatures(Query q, Document doc, double score, Map<String, Double> idfs) {
		// TODO: we should use BM25 to calculate tf!
		Map<String, Map<String, Double>> tfs = getDocTermFreqs(doc, q);
		normalizeTFs(tfs, doc, q);

		Map<String, Double> tfQuery = getQueryFreqs(q, idfs);

        double[] instance = new double[6];
        
        instance[0] = dotProduct(tfQuery, tfs.get("url"));
        instance[1] = dotProduct(tfQuery, tfs.get("title"));
        instance[2] = dotProduct(tfQuery, tfs.get("body"));
        instance[3] = dotProduct(tfQuery, tfs.get("header"));
        instance[4] = dotProduct(tfQuery, tfs.get("anchor"));
        instance[5] = score;

        return instance;
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
}
