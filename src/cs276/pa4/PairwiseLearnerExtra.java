package cs276.pa4;

import static cs276.pa4.Util.loadRelData;
import static cs276.pa4.Util.loadTrainData;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cs276.utils.MapUtils;
import cs276.utils.Pair;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class PairwiseLearnerExtra extends Learner {
  private LibSVM model;
  public PairwiseLearnerExtra(boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
  public PairwiseLearnerExtra(double C, double gamma, boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    
    model.setCost(C);
    model.setGamma(gamma); // only matter for RBF kernel
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {
		
		// Return the training data as a matrix
		TestFeatures testFeatures = extractAllFeatures("train_dataset", train_data_file, train_rel_file, idfs);
		return convertToPairwise(testFeatures);
	}

	@Override
	public Classifier training(Instances dataset) {
		
		try {
			model.buildClassifier(dataset);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return model;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) {
		return extractAllFeatures("test_dataset", test_data_file, null, idfs);
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) {
		
		Map<String, List<String>> results = new HashMap<String, List<String>>();
		
		// Standardize features
		standardizeInstances(tf);
		
		for (String query : tf.index_map.keySet()) {
			Map<String, Integer> candidates = tf.index_map.get(query);
			List<String> rankings = rankDocumentsPairwise(query, candidates, model, tf.features);
			results.put(query, rankings);
		}
		return results;
	}
	
	/**
	 * Standardizes the Instances in tf to have a mean of 0 and a 
	 * standard deviation of 1.
	 * 
	 * @param tf Test features with input data set
	 */
	private void standardizeInstances(TestFeatures tf) {

		// TODO make sure the order of the vectors stays the same
		try {
			Standardize filter = new Standardize();
			filter.setInputFormat(tf.features);
			Instances standardized = Filter.useFilter(tf.features, filter);	
			tf.features = standardized;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}
	
	private Instances convertToPairwise(TestFeatures pointwiseFeatures) {
		
		// Perform standardization
		standardizeInstances(pointwiseFeatures);
		
		// Take differences to create pairwise training examples
		Instances features = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		
		// Addition features
		attributes.add(new Attribute("bm25_score"));
		attributes.add(new Attribute("pagerank"));
		
		/*
		 * TODO Add new features here
		 */
		
		// Create attributes of class labels
		ArrayList<String> classLabels = new ArrayList<>();
		classLabels.addAll(Arrays.asList(new String[]{"1", "-1"}));
		attributes.add(new Attribute("classification", classLabels));

		features = new Instances("pairwise_training_features", attributes, 0);
		features.setClassIndex(attributes.size() - 1);
		
		// Add each pair to the features
		for(String query : pointwiseFeatures.index_map.keySet()) {
			
			Map<String, Integer> docs = pointwiseFeatures.index_map.get(query);
			
			List<Pair<String, Integer>> pairs = MapUtils.convertToPairs(docs);
			
			// Sort the docs by their relevance score
			Collections.sort(pairs, Collections.reverseOrder(new Comparator<Pair<String,Integer>>(){
				@Override
				public int compare(Pair<String, Integer> o1, Pair<String, Integer> o2) {
					Instance inst1 = pointwiseFeatures.features.get(o1.getSecond());
					Instance inst2 = pointwiseFeatures.features.get(o2.getSecond());
				
					return (int) (inst1.value(inst1.numAttributes() - 1) - inst2.value(inst1.numAttributes() - 1));
				}	
			}));
			
			// Boolean to decide whether to compute a - b or b - a
			// This ensure that there are roughly the same number of positive and negative examples
			boolean reversePair = false;
			
			for (int i = 0; i < pairs.size(); i++) {
				
				for (int j = i; j < pairs.size(); j++) {
					
					// Get first vector
					Instance inst1 = pointwiseFeatures.features.get(pairs.get(i).getSecond());
					
					// Get second vector
					Instance inst2 = pointwiseFeatures.features.get(pairs.get(j).getSecond());
					
					// If the relevance ratings are not the same then use the pair as a training example
					if (inst1.value(inst1.numAttributes() - 1) != inst2.value(inst1.numAttributes() - 1)) {
						
						double[] inst1vec = inst1.toDoubleArray();
						double[] inst2vec = inst2.toDoubleArray();
						
						// Copy everything but relevance ranking into feature vector
						double[] fv1 = new double[inst1vec.length - 1];
						double[] fv2 = new double[inst2vec.length - 1];
						
						for (int k = 0; k < fv1.length; k++) {
							fv1[k] = inst1vec[k];
							fv2[k] = inst2vec[k];
						}
						
						Instance fv;
						
						if (reversePair) {
							// Compute difference
							double [] difference = computeDifference(fv2, fv1);
							
							// Make feature vector with correct classification
							fv = new DenseInstance(difference.length + 1);
							fv.setDataset(features);
							
							for (int k = 0; k < difference.length; k++) {
								fv.setValue(k, difference[k]);
							}
							
							fv.setValue(difference.length, "-1");

						} else {
							// Compute difference
							double [] difference = computeDifference(fv1, fv2);
							
							// Make feature vector with correct classification
							fv = new DenseInstance(difference.length + 1);
							fv.setDataset(features);
							
							for (int k = 0; k < difference.length; k++) {
								fv.setValue(k, difference[k]);
							}
							
							fv.setValue(difference.length, "1");
						}

						// Add new feature vector
						features.add(fv);
						
						reversePair = !reversePair;
					}
				}
			}
		}
		
		return features;
	}
	
	private List<String> rankDocumentsPairwise(String query,
			Map<String, Integer> candidates, Classifier model, Instances features) {
		
		List<Pair <String, Integer>> pairs = MapUtils.convertToPairs(candidates);
		
		// Sort using scorer comparator
		Collections.sort(pairs, Collections.reverseOrder(new Comparator<Pair<String,Integer>>(){
			@Override
			public int compare(Pair<String, Integer> o1, Pair<String, Integer> o2) {
				Instance inst1 = features.get(o1.getSecond());
				Instance inst2 = features.get(o2.getSecond());
			
				return compareInstances(inst1, inst2, model, features);
			}	
		}));
		
		
		List<String> ranked = new ArrayList<String>();
		
		for (Pair pair : pairs) {
			ranked.add((String) pair.getFirst());
		}
		return ranked;
	}
	
	private int compareInstances(Instance inst1, Instance inst2, Classifier model, Instances features) {
		
		// Constuct vector corresponding to the difference between the two instances
		double[] inst1vec = inst1.toDoubleArray();
		double[] inst2vec = inst2.toDoubleArray();
		
		double[] difference = computeDifference(inst1vec, inst2vec);
		
		// Classify the difference vector
		Instance diffInst = new DenseInstance(1.0, difference);
		diffInst.setDataset(features);
		
		double indexClassification = 0;
		try {
			indexClassification = model.classifyInstance(diffInst);
			
			//System.out.println(features.classAttribute().value((int) indexClassification));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		if (indexClassification > 0) {
			return 1;
		} else {
			return -1;
		}
		
	}
	
	private double[] computeDifference(double [] a, double [] b) {
		
		double [] difference = new double[a.length];
		
		for (int i = 0; i < a.length; i++) {
			difference[i] = a[i] - b[i];
		}
		return difference;
	}

}
