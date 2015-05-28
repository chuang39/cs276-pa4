package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import cs276.pa4.Query;
import cs276.utils.Pair;

public class SmallestWindowScorer extends AScorer {

	public SmallestWindowScorer(Map<String, Double> idfs) {
		super(idfs);
		// TODO Auto-generated constructor stub
	}

	static Integer getPostingValueByPair(Map<String, List<Integer>> d, Pair<Integer, String> p) {
		// Pair's first value is the index and seconds value is the word of posting list.
		return d.get(p.getSecond()).get(p.getFirst());
	}
	
	static public double checkWindowBody(Query q, Map<String, List<Integer>> d) {
		int querylen = q.words.size();
		int doclen = d.size();
		if (querylen != doclen) {
			return Double.MAX_VALUE;
		}

		Comparator<Pair<Integer, String>> comparator = new Comparator<Pair<Integer, String>>() {
			@Override
			public int compare(Pair<Integer, String> x, Pair<Integer, String> y)
			{
			    return getPostingValueByPair(d, x).compareTo(getPostingValueByPair(d, y));
			}
		};
		PriorityQueue<Pair<Integer, String>> pq = new PriorityQueue<Pair<Integer, String>>(doclen, comparator);

		int curMax = 0;
		int winSize = Integer.MAX_VALUE;
		for (String s : d.keySet()) {
			curMax = Math.max(curMax, d.get(s).get(0));
			Pair<Integer, String> p = new Pair<Integer, String>(0, s);
			pq.add(p);
		}

		while (pq.size() == doclen) {
			Pair<Integer, String> curMinPair = pq.poll();
			winSize = Math.min(winSize, curMax - getPostingValueByPair(d, curMinPair)+1);
			String curString = curMinPair.getSecond();
			Integer curIndex= curMinPair.getFirst();
			if (d.get(curString).size()-1 > curIndex) {
				Pair<Integer, String> newPair = new Pair<Integer, String>(curIndex+1, curString);
				pq.add(newPair);
				curMax = Math.max(curMax, getPostingValueByPair(d, newPair));
			}

		}
		return winSize;
	}

	static public double checkWindowNonBody(Query q,String docstr,double curSmallestWindow) {
		/*
		 * Find the minimum window size of docstr for query q.
		 */
		List<String> queryWords = q.words;
		List<String> docWords = new ArrayList<String>(Arrays.asList(docstr.split(" ")));
		Map<String, Integer> needFind = new HashMap<String, Integer>();	// key is the needed string, value is the number it is needed in query
	    Map<String, Integer> hasFound = new HashMap<String, Integer>();	// key is the needed string, value is the number it is found in sequence

	    // Initialize two hashmaps which are used to fine the minimum sequence window
	    for (int i = 0; i < queryWords.size(); i++) {
	    	hasFound.put(queryWords.get(i), 0);
            if (needFind.containsKey(queryWords.get(i))) {
            	needFind.put(queryWords.get(i), needFind.get(queryWords.get(i))+1);
            } else {
                needFind.put(queryWords.get(i), 1);
            }
        }

	    ArrayList<Integer> nexts = new ArrayList<Integer>();
        int right = 0, left = 0, found = 0; // notice here: right points to S while left points to next[]. next[] holds real index of S.

        //String window = "";
        double winSize = curSmallestWindow;
        while (right < docWords.size()) {
            String s = docWords.get(right);
            if (!needFind.containsKey(s)) {     // We don't need this word in doc
                right++;
                continue;
            }

            nexts.add(right); right++;
            hasFound.put(s, hasFound.get(s)+1);
            if (hasFound.get(s) <= needFind.get(s)) found++;    // we found necessary char; otherwise, it is useful, but not necessary at this point

            if (found >= queryWords.size()) {    // got a window
                // Check how far we can move the left
                String leftString = docWords.get(nexts.get(left));
                while (hasFound.get(leftString) > needFind.get(leftString)) {
                    hasFound.put(leftString, hasFound.get(leftString)-1);
                    left++;
                    leftString = docWords.get(nexts.get(left));
                }
                if (right - nexts.get(left) < winSize) {
                    winSize = right - nexts.get(left);
                    //window = docWords[nexts.get(left), right]
                }
            }
        }


		return winSize;
	}

	@Override
	public List<Double> getSimScore(Document d, Query q) {
		List<Double> res = new ArrayList<Double>();
		double query_len = q.words.size();
		
		double url_window = 0;
		if (d.url != null) {
			double url_window_length = checkWindowNonBody(q, d.url, Double.MAX_VALUE);
			url_window = (url_window_length >= 50) ? 0 : query_len / url_window_length;
		}

		double title_window = 0;
		if (d.title != null) {
			double title_window_length = checkWindowNonBody(q, d.title, Double.MAX_VALUE);
			title_window = (title_window_length >= 50) ? 0 : query_len / title_window_length;
		}
		
		double body_window = 0;
		if (d.body_hits != null && !d.body_hits.isEmpty()) {
			double body_window_length = checkWindowBody(q, d.body_hits);
			body_window = (body_window_length >= 50) ? 0 : query_len / body_window_length;
		}

		double header_window = 0;
		if (d.headers != null && !d.headers.isEmpty()) {
			double header_window_length = Double.MAX_VALUE;
			for (String s : d.headers) {
				header_window_length = checkWindowNonBody(q, s, header_window_length);
			}
			header_window = (header_window_length >= 50) ? 0 : query_len / header_window_length;
		}
		
		double anchor_window = 0;
		if (d.anchors != null && !d.anchors.isEmpty()) {
			double anchor_window_length = Double.MAX_VALUE;
			for (String s : d.anchors.keySet()) {
				anchor_window_length = checkWindowNonBody(q, s, anchor_window_length);
			}
			anchor_window = (anchor_window_length >= 50) ? 0 : query_len / anchor_window_length;
		}
		
		res.add(url_window);
		res.add(title_window);
		res.add(body_window);
		res.add(header_window);
		res.add(anchor_window);

		return res;
	}

}
