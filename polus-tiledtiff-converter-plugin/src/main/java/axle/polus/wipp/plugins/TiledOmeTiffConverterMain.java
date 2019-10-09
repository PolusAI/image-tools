package axle.polus.wipp.plugins;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.logging.Level;
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
import loci.formats.ImageReader;

public class TiledOmeTiffConverterMain {

	private static final Logger LOG = Logger.getLogger(
			TiledOmeTiffConverterMain.class.getName());
	
	public static void main(String[] args) throws IOException, Exception {
		// sanity checks
		int i = 0;
		LOG.log(Level.INFO, "argument length=" + args.length);
		for (String arg : args) LOG.log(Level.INFO, "args[" + i++ + "]:" + arg);

		// set up options parser
		Options options = new Options();

		Option input = new Option("i", "input", true, "input folder");
		input.setRequired(true);
		options.addOption(input);

		Option output = new Option("o", "output", true, "output folder");
		output.setRequired(true);
		options.addOption(output);

		Option tileSizeX = new Option("xtile", "tileSizeX", true, "The tile width.");
		tileSizeX.setRequired(true);
		options.addOption(tileSizeX);

		Option tileSizeY = new Option("ytile", "tileSizeY", true, "The tile height.");
		tileSizeY.setRequired(true);
		options.addOption(tileSizeY);
		
		Option useMetadata = new Option("useMeta", "useMetadata", true, "Convert metadata to tiled tiff.");
		useMetadata.setRequired(true);
		options.addOption(useMetadata);

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
		
		boolean useMeta = Boolean.valueOf(cmd.getOptionValue("useMetadata"));

		String inputFileDir = cmd.getOptionValue("input");
		if (useMeta) {
			inputFileDir = inputFileDir.replace("/images", "/metadata_files");
		}
		String outputFileDir = cmd.getOptionValue("output");

		LOG.log(Level.INFO, "inputFileDir=" + inputFileDir);
		LOG.log(Level.INFO, "outputFileDir=" + outputFileDir);

		String tileSizeXStr = cmd.getOptionValue("tileSizeX");
		String tileSizeYStr = cmd.getOptionValue("tileSizeY");

		int tileSizeXPix = Integer.valueOf(tileSizeXStr);
		int tileSizeYPix = Integer.valueOf(tileSizeYStr);

		File inputFolder = new File (inputFileDir);
		File outputFolder = new File (outputFileDir);
		
		if((tileSizeXPix % 16) != 0 || tileSizeXPix < 16){
			LOG.log(Level.SEVERE, "tileSizeX must be positive and a multiple of 16.");
			System.err.println("ERROR: tileSizeX must be positive and a multiple of 16.");
			return;
		}
		
		if((tileSizeYPix % 16) != 0 || tileSizeYPix < 16){
			LOG.log(Level.SEVERE, "tileSizeY must be positive and a multiple of 16.");
			System.err.println("ERROR: tileSizeY must be positive and a multiple of 16.");
			return;
		}

		LOG.log(Level.INFO, "tileSizeXPix=" + tileSizeXPix);
		LOG.log(Level.INFO, "tileSizeYPix=" + tileSizeYPix);
		
        File[] images =  inputFolder.listFiles(new FilenameFilter() {
        	String[] suffixes = (new ImageReader()).getSuffixes();
        	
			public boolean accept(File dir, String name) {
				for (String ftype : suffixes) {
					if (name.toLowerCase().endsWith(ftype.toLowerCase())) {
						return true;
					}
				}
				LOG.log(Level.INFO, "File is not supported by Bioformats and will be skipped: " + name);
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
		
		LOG.log(Level.INFO, "Starting tile tiff converter!!");
		
		for (File image : images) {
			String outFile = outputFileDir.concat(File.separator).concat(FilenameUtils.getBaseName(image.getName())).concat(".ome.tif");
			
			TiledOmeTiffConverter tiledReadWriter = new TiledOmeTiffConverter(image.getAbsolutePath(), outFile, tileSizeXPix, tileSizeYPix);
			// initialize the files
			tiledReadWriter.run();
			
		}

		int exitVal = 0;
		String err = "";
		
		if (exitVal != 0){
			throw new RuntimeException(err);
		}
		LOG.info("The end of tile tiff conversion!!");  
	}
}
