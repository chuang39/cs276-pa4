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
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import static cs276.pa4.Util.loadTrainData;
import static cs276.pa4.Util.loadRelData;
import static cs276.utils.MapUtils.convertToPairs;

public class PointwiseLearnerExtra extends Learner {
	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {
		TestFeatures testFeatures = extractAllFeatures("train_dataset", train_data_file, train_rel_file, idfs);
        return testFeatures.features;
	}

	@Override
	public Classifier training(Instances dataset) {
		LibSVM svm = new LibSVM();
        svm.setSVMType(new SelectedTag(LibSVM.SVMTYPE_NU_SVR, LibSVM.TAGS_SVMTYPE));
       // svm.setCost(50);
       // svm.setNu(0.05);
        svm.setShrinking(false);

        try {
            svm.buildClassifier(dataset);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return svm;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) {
		return extractAllFeatures("test_dataset", test_data_file, null, idfs);
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
}
