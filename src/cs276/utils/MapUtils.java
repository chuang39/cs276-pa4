package cs276.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;



public class MapUtils {
	public static <K, V> List<Pair<K, V>> convertToPairs(Map<K, V> map) {
		List<Pair<K, V>> res = new ArrayList<Pair<K, V>>();
		
		for (Map.Entry<K, V> entry : map.entrySet()) {
			Pair<K, V> newPair = new Pair<K, V>(entry.getKey(), entry.getValue());
			res.add(newPair);
		}
		
		return res;
	}
}
