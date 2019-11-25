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
import loci.common.services.DependencyException;
import loci.common.services.ServiceException;
import loci.formats.FormatException;
import loci.formats.ImageReader;

public class TiledOmeTiffConverterMain {

	private static final Logger LOG = Logger.getLogger(
			TiledOmeTiffConverterMain.class.getName());
	
	/**
	 * WIPP Plugin entrance point
	 * 
	 * This function handles arguments passed in from the command line, finds all Bioformats compatible
	 * input files, and starts a tiled tiff conversion thread for each file.
	 * 
	 * Note: Bioformats supports importing images from txt, csv, and excel spreadsheet formats. This
	 * plugin deliberately excludes txt, csv, and excel spreadsheet formats from conversion.
	 * 
	 * @param args
	 * @throws IOException
	 * @throws Exception
	 */
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
		
		Option useMetadata = new Option("useMeta", "useMetadata", true, "Convert metadata to tiled tiff.");
		useMetadata.setRequired(false);
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
		
		boolean useMeta;
		if (cmd.getOptionValue("useMetadata")==null) {
			useMeta = false;
		} else {
			useMeta = Boolean.valueOf(cmd.getOptionValue("useMetadata"));
		}

		String inputFileDir = cmd.getOptionValue("input");
		if (useMeta) {
			inputFileDir = inputFileDir.replace("/images", "/metadata_files");
		}
		String outputFileDir = cmd.getOptionValue("output");

		LOG.log(Level.INFO, "inputFileDir=" + inputFileDir);
		LOG.log(Level.INFO, "outputFileDir=" + outputFileDir);

		int tileSizeXPix = 1024;
		int tileSizeYPix = 1024;

		File inputFolder = new File (inputFileDir);
		File outputFolder = new File (outputFileDir);

		LOG.log(Level.INFO, "tileSizeXPix=" + tileSizeXPix);
		LOG.log(Level.INFO, "tileSizeYPix=" + tileSizeYPix);
		
        File[] images =  inputFolder.listFiles(new FilenameFilter() {
        	String[] suffixes = (new ImageReader()).getSuffixes();
        	
			public boolean accept(File dir, String name) {
				for (String ftype : suffixes) {
					if (name.toLowerCase().endsWith("csv") || 
							name.toLowerCase().endsWith("txt") || 
							name.toLowerCase().endsWith("xlsx") ||
							name.toLowerCase().endsWith("xls")) {
						LOG.log(Level.INFO, "File will not be converted to tiled tiff: " + name);
						return false;
					}
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
			
			Thread thread = new Thread() {
				public void run() {
					String outFile = outputFileDir.concat(File.separator).concat(FilenameUtils.getBaseName(image.getName())).concat(".ome.tif");
					
					TiledOmeTiffConverter tiledReadWriter = new TiledOmeTiffConverter(image.getAbsolutePath(), outFile, tileSizeXPix, tileSizeYPix);
					
					// initialize the files
					try {
						tiledReadWriter.run();
					} catch (FormatException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					} catch (DependencyException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					} catch (ServiceException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			};
			
			thread.start();
		}

		int exitVal = 0;
		String err = "";
		
		if (exitVal != 0){
			throw new RuntimeException(err);
		}
		LOG.info("The end of tile tiff conversion!!");  
	}
}
