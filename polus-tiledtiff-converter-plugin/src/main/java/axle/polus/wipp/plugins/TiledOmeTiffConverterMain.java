package axle.polus.wipp.plugins;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.ConsoleHandler;
import java.util.logging.Formatter;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.logging.Formatter;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.io.FilenameUtils;

import axle.polus.data.utils.converters.TiledOmeTiffConverter;
import loci.common.DebugTools;
import loci.common.services.ServiceFactory;
import loci.formats.ImageReader;
import loci.formats.meta.IMetadata;
import loci.formats.services.OMEXMLService;

public class TiledOmeTiffConverterMain {

	private static final Logger LOG = Logger.getLogger(
			TiledOmeTiffConverterMain.class.getName());
	
	/**
	 * WIPP Plugin entrance point
	 * 
	 * This function handles arguments passed in from the command line, finds
	 * all Bioformats compatible input files, and starts a tiled tiff conversion
	 * thread for each file.
	 * 
	 * Note: Bioformats supports importing images from txt, csv, and excel
	 * spreadsheet formats. This plugin deliberately excludes txt, csv, and
	 * excel spreadsheet formats from conversion.
	 * 
	 * @param args
	 * @throws IOException
	 * @throws Exception
	 */
	public static void main(String[] args) throws IOException, Exception {
		/*
		 * Plugin Initialization
		 */
		
		// Setup the thread logger
		Logger TLOG = Logger.getLogger(TiledOmeTiffConverter.class.getName());
		TLOG.setUseParentHandlers(false);

		// Create a custom handler so the output looks pretty
		CustomLogger customFormatter = new CustomLogger();
		ConsoleHandler handler = new ConsoleHandler();
		handler.setFormatter(customFormatter);

		TLOG.addHandler(handler);
		
		// set Bioformats logger level to avoid standard errors
		DebugTools.setRootLevel("error");
		
		// sanity checks
		int i = 0;
		LOG.log(Level.INFO, "argument length=" + args.length);
		for (String arg : args) LOG.log(Level.INFO, "args[" + i++ + "]:" + arg);

		/*
		 * Parse the input values
		 */
		// set up options parser
		Options options = new Options();

		Option input = new Option("i", "input", true, "input folder");
		input.setRequired(true);
		options.addOption(input);

		Option output = new Option("o", "output", true, "output folder");
		output.setRequired(true);
		options.addOption(output);

		// parse the options
		CommandLineParser parser = new DefaultParser();
		HelpFormatter formatter = new HelpFormatter();
		CommandLine cmd;

		try {
			cmd = parser.parse(options, args);
		} catch (ParseException e) {
			LOG.log(Level.SEVERE,e.getMessage());
			formatter.printHelp("utility-name", options);

			System.exit(1);
			return;
		}

		String inputFileDir = cmd.getOptionValue("input");
		String outputFileDir = cmd.getOptionValue("output");

		LOG.info("inputFileDir=" + inputFileDir);
		LOG.info("outputFileDir=" + outputFileDir);

		// Set tile sizes as a variable to easily change it in the future
		int tileSizeXPix = 1024;
		int tileSizeYPix = 1024;

		File inputFolder = new File (inputFileDir);
		File outputFolder = new File (outputFileDir);

		LOG.info("tileSizeXPix=" + tileSizeXPix);
		LOG.info("tileSizeYPix=" + tileSizeYPix);
		
		/*
		 * Parse files, skipping over txt, csv, xls, and xlsx file types
		 */
        File[] images =  inputFolder.listFiles(new FilenameFilter() {
        	String[] suffixes = (new ImageReader()).getSuffixes();
        	
			public boolean accept(File dir, String name) {
				for (String ftype : suffixes) {
					if (name.toLowerCase().endsWith("csv") || 
							name.toLowerCase().endsWith("txt") || 
							name.toLowerCase().endsWith("xlsx") ||
							name.toLowerCase().endsWith("xls")) {
						LOG.info("File will not be converted to tiled tiff: " + name);
						return false;
					}
					if (name.toLowerCase().endsWith(ftype.toLowerCase())) {
						return true;
					}
				}
				LOG.info("File is not supported by Bioformats and will be skipped: " + name);
				return false;
			}
		});

        if (images == null) {
        	throw new NullPointerException("Input folder is empty");
        }

        boolean created = outputFolder.mkdirs();
        if (!created && !outputFolder.exists()) {
            throw new IOException("Can not create folder " + outputFolder);
        }
		
        /*
         * Start file conversion threads
         */
        // Create one thread for every two cores, up to 16 threads (min 1)
        int p = Math.max(1,Math.min(16,Runtime.getRuntime().availableProcessors()/2));
        ExecutorService pool = Executors.newFixedThreadPool(p);
		LOG.log(Level.INFO, "Starting tile tiff converter with " + p + " threads...");
        
		// Create the threads and run
        for (File image : images) {
			String baseFile = outputFileDir.concat(File.separator).concat(FilenameUtils.getBaseName(image.getName())).concat(".ome.tif");

			// construct the object that stores OME-XML metadata
			ServiceFactory factory = new ServiceFactory();
			OMEXMLService service = factory.getInstance(OMEXMLService.class);
			IMetadata omexml = service.createOMEXMLMetadata();

			ImageReader reader = new ImageReader();
			reader.setOriginalMetadataPopulated(true);
			reader.setMetadataStore(omexml);
			reader.setId(image.getAbsolutePath());

			// Get the width of file name variables
			int width = Math.max(reader.getSizeC(), reader.getSizeT());
			width = Math.max(reader.getSizeZ(), width);
			width = (int) Math.ceil(Math.log10(width));

			for (int z = 0; z < reader.getSizeZ(); ++z) {
				for (int c = 0; c < reader.getSizeC(); ++c) {
					for (int t = 0; t < reader.getSizeT(); ++t) {
						
						String outFile = baseFile;
						if (reader.getSizeC()>1) {
							outFile = outFile.replace(".ome.tif", "_c" + String.format("%0" + width + "d", c) + ".ome.tif");
						}
						if (reader.getSizeT()>1) {
							outFile = outFile.replace(".ome.tif", "_t" + String.format("%0" + width + "d", t) + ".ome.tif");
						}
						if (reader.getSizeZ()>1) {
							outFile = outFile.replace(".ome.tif", "_z" + String.format("%0" + width + "d", z) + ".ome.tif");
						}
						
						TiledOmeTiffConverter tiledReadWriter = new TiledOmeTiffConverter(reader, outFile, z, c, t);
						pool.execute(tiledReadWriter);
					}
				}
			}

			reader.close();
        }

		int exitVal = 0;
		String err = "";
		
		if (exitVal != 0){
			throw new RuntimeException(err);
		}
		
		pool.shutdown();
		LOG.info("The end of tile tiff conversion!!");  
	}
}

class CustomLogger extends Formatter {

	private static final DateFormat df = new SimpleDateFormat("dd/MM/yyyy hh:mm:ss.SSS");

    /* (non-Javadoc)
    * @see java.util.logging.Formatter#format(java.util.logging.LogRecord)
    */
    @Override
    public String format(LogRecord record) {
        StringBuilder sb = new StringBuilder();
		sb.append(df.format(new Date(record.getMillis()))).append(" - ");
        sb.append(record.getLevel()).append(":");
        sb.append(record.getMessage()).append('\n');
        return sb.toString();
    }    
}