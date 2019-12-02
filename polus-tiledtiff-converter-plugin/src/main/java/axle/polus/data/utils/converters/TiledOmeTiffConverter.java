package axle.polus.data.utils.converters;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.javatuples.Triplet;

import loci.common.DebugTools;
import loci.common.services.DependencyException;
import loci.common.services.ServiceException;
import loci.common.services.ServiceFactory;
import loci.formats.FormatException;
import loci.formats.FormatTools;
import loci.formats.ImageReader;
import loci.formats.codec.CompressionType;
import loci.formats.meta.IMetadata;
import loci.formats.out.OMETiffWriter;
import loci.formats.services.OMEXMLService;
import ome.xml.model.primitives.PositiveInteger;

/**
 * Based off of the example from https://docs.openmicroscopy.org/bio-formats/5.9.1/developers/tiling.html
 *
 * @author nick.schaub at nih.gov
 */
public class TiledOmeTiffConverter {
	/**
	 * Class that handles conversion of any Bioformats supported image format into an OME tiled tiff.
	 */

	private static final Logger LOG = Logger.getLogger(TiledOmeTiffConverter.class.getName());

	private ImageReader reader;
	private ArrayList<Triplet<Integer,Integer,OMETiffWriter>> fileWriters;
	private String inputFile;
	private String outputFile;
	private int tileSizeX;
	private int tileSizeY;

	/**
	 * Class constructor
	 * 
	 * @param inputFile	Complete file path to image that will be converted
	 * @param outputFile	Complete path to export folder
	 * @param tileSizeX	Tile size width, generally 1024
	 * @param tileSizeY	Tile size height, generally 1024
	 */
	public TiledOmeTiffConverter(String inputFile, String outputFile, int tileSizeX, int tileSizeY) {
		this.inputFile = inputFile;
		this.outputFile = outputFile;
		this.tileSizeX = tileSizeX;
		this.tileSizeY = tileSizeY;
	}

	/**
	 * Initialize conversion process
	 * 
	 * This function reads basic metadata from the input file and initializes all output files.
	 * Each output file will contain at most three dimensions: X,Y,Z. If multiple channels or
	 * time-points are present within the input file, multiple output files will be initialized. 
	 * 
	 * @throws DependencyException
	 * @throws FormatException
	 * @throws IOException
	 * @throws ServiceException
	 */
	private void init() throws DependencyException, FormatException, IOException, ServiceException {
		DebugTools.setRootLevel("error");

		// construct the object that stores OME-XML metadata
		ServiceFactory factory = new ServiceFactory();
		OMEXMLService service = factory.getInstance(OMEXMLService.class);
		IMetadata omexml = service.createOMEXMLMetadata();

		// set up the reader and associate it with the input file
		reader = new ImageReader();
		reader.setOriginalMetadataPopulated(true);
		reader.setMetadataStore(omexml);
		reader.setId(inputFile);

		// Generate file writer objects for each z, c, and t
		fileWriters = new ArrayList<Triplet<Integer,Integer,OMETiffWriter>>();
		for (int c=0; c<reader.getSizeC(); c++) {
			String outFileC = outputFile;
			if (reader.getSizeC()>1) {
				outFileC = outFileC.replace(".ome.tif", "_c" + String.format("%03d", c) + ".ome.tif");
			}
			for (int t=0; t<reader.getSizeT(); t++) {
				String outFileCT = outFileC;
				if (reader.getSizeT()>1) {
					outFileCT = outFileCT.replace(".ome.tif", "_t" + String.format("%03d", t) + ".ome.tif");
				}
				if (reader.getSizeZ()>1) {
					outFileCT = outFileCT.replace(".ome.tif", "_z<000-" + String.format("%03d", reader.getSizeZ()) + ">.ome.tif");
				}
				// important to delete because OME uses RandomAccessFile
				Path outputPath = Paths.get(outFileCT);
				outputPath.toFile().delete();

				// set up the writer and associate it with the output file
				OMETiffWriter writer = new OMETiffWriter();
				writer.setMetadataRetrieve(omexml);
				
			    // set the tile size height and width for writing
			    this.tileSizeX = writer.setTileSizeX(tileSizeX);
			    this.tileSizeY = writer.setTileSizeY(tileSizeY);

				writer.setId(outFileCT);

				writer.setCompression(CompressionType.LZW.getCompression());
				
				LOG.info(reader.toString() + ": Initializing writer for slice: (c=" + Integer.toString(c) + ",t=" + Integer.toString(t) + ")");
				
				Triplet<Integer,Integer,OMETiffWriter> quartet = new Triplet<Integer,Integer,OMETiffWriter>(c, t, writer);	
				fileWriters.add(quartet);
			}
		}
	}

	/**
	 * Read input file and write output file in tiles
	 * 
	 * The input files is read in tiles of size (tileSizeX, tileSizeY), with only one z-slice at a time.
	 * Each channel and time-point (if present) are saved to separate files.
	 * 
	 * It is important to save images in tiles, since Bioformats indexes pixels using unsigned integers.
	 * This means that loading a full image plane larger than 2GB will throw an indexing error. Saving
	 * in tiles also has the benefit of being memory efficient, so it can be run on small nodes.
	 * 
	 * @throws FormatException
	 * @throws DependencyException
	 * @throws ServiceException
	 * @throws IOException
	 */
	public void readWriteTiles() throws FormatException, DependencyException, ServiceException, IOException {
		int bpp = FormatTools.getBytesPerPixel(reader.getPixelType());
		int tilePlaneSize = tileSizeX * tileSizeY * reader.getRGBChannelCount() * bpp;
		byte[] buf = new byte[tilePlaneSize];

		// convert each image in the current series
		for (int channel=0; channel < reader.getSizeC(); channel++) {
			for (int timepoint=0; timepoint < reader.getSizeT(); timepoint++) {
				OMETiffWriter writer = null;
				LOG.info(reader.toString() + ": Writing slice: (c=" + Integer.toString(channel) + ",t=" + Integer.toString(timepoint) + ")");
				for (int w = 0; w<fileWriters.size(); w++) {
					Triplet<Integer, Integer, OMETiffWriter> triplet = fileWriters.get(w);
					if (triplet.getValue0()==channel && triplet.getValue1()==timepoint) {
						writer = triplet.getValue2();
						break;
					}
				}
				
				int width = reader.getSizeX();
				int height = reader.getSizeY();

				// Determined the number of tiles to read and write
				int nXTiles = width / tileSizeX;
				int nYTiles = height / tileSizeY;
				if (nXTiles * tileSizeX != width) nXTiles++;
				if (nYTiles * tileSizeY != height) nYTiles++;

				IMetadata omexml = (IMetadata) writer.getMetadataRetrieve();
				omexml.setPixelsSizeZ(new PositiveInteger(reader.getSizeZ()), 0);
				omexml.setPixelsSizeC(new PositiveInteger(1), 0);
				omexml.setPixelsSizeT(new PositiveInteger(1), 0);
				
				for (int z=0; z<reader.getSizeZ(); z++) {
					int index = reader.getIndex(z, channel, timepoint);
					for (int y=0; y<nYTiles; y++) {
						for (int x=0; x<nXTiles; x++) {
							
							int tileX = x * tileSizeX;
							int tileY = y * tileSizeY;
							
							int effTileSizeX = (tileX + tileSizeX) < width ? tileSizeX : width - tileX;
							int effTileSizeY = (tileY + tileSizeY) < height ? tileSizeY : height - tileY;

							buf = reader.openBytes(index, tileX, tileY, effTileSizeX, effTileSizeY);
							writer.saveBytes(z, buf, tileX, tileY, effTileSizeX, effTileSizeY);
						}
					}
				}
				try {
					writer.close();
				}
				catch (IOException e) {
					LOG.log(Level.WARNING, reader.toString() + ": Failed to close writers.",e);
				}
			}
		}
	}

	/**
	 * Close open file readers
	 */
	private void cleanup() {
		try {
			reader.close();
		}
		catch (IOException e) {
			LOG.log(Level.WARNING, reader.toString() + ": Failed to close reader.",e);
		}
	}

	/**
	 * Main process, set up for threading
	 * 
	 * @throws FormatException
	 * @throws IOException
	 * @throws DependencyException
	 * @throws ServiceException
	 */
	public void run() throws FormatException, IOException, DependencyException, ServiceException {
		this.init();

		try {
			// read and write the image using tiles
			this.readWriteTiles();
		}
		catch(Exception e) {
			System.err.println("Failed to read and write tiles.");
			e.printStackTrace();
			throw e;
		}
		finally {
			// close the files
			this.cleanup();
		}
	}

}
