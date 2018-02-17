import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.PrintWriter;

public class TextConvert {

	public static void main(String[] args) {
		try {
			BufferedReader in = new BufferedReader(new FileReader(args[0]));
			PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("out_"+args[0])));
			String line = in.readLine();
			StringBuilder sb = new StringBuilder();
			while (line != null) {
				int idx = line.indexOf(" ");
				sb.append(line.substring(0,idx));
				sb.append(",");
				sb.append(line.substring(idx+1,line.length()));
				sb.append(",?");
				out.println(sb.toString());
				sb.setLength(0);
				line = in.readLine();
			}
			out.close();
			in.close();
		}
		catch (IOException ioex) { ioex.printStackTrace(); }
	}
	
}