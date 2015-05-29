package cs276.pa4;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class pdfDetectionScorer {
	boolean checkPdf(String url) {
		if (url != null && url.length() > 4) {
			String lastFourChars = url.substring(url.length()-4);
			if (lastFourChars.equals(".pdf")) {
				return true;
			}
		}
		return false;
	}
	
	public List<Double> getSimScore(Document d) {
		List<Double> res = new ArrayList<Double>();
		
		if (checkPdf(d.url)) {
			res.add(1.0);
		} else {
			res.add(0.0);
		}
		
		return res;
	}

	public static void main(String [] args) {
		pdfDetectionScorer pds = new pdfDetectionScorer();
		System.out.println(pds.checkPdf("http://asbadf.pdf"));
		System.out.println(pds.checkPdf("http://asbadf/pdf"));
		System.out.println(pds.checkPdf("http://asbadfspdf"));
		System.out.println(pds.checkPdf(".pdf"));
	}
}
