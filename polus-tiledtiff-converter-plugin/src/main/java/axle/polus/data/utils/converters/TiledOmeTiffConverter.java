package axle.polus.data.utils.converters;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.javatuples.Quartet;

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

/**
 * Based off of the example from https://docs.openmicroscopy.org/bio-formats/5.9.1/developers/tiling.html
 *
 * @author nick.schaub at nih.gov
 */
public class TiledOmeTiffConverter {

	private static final Logger LOG = Logger.getLogger(TiledOmeTiffConverter.class.getName());

	private ImageReader reader;
	private ArrayList<Quartet<Integer,Integer,Integer,OMETiffWriter>> fileWriters;
	private String inputFile;
	private String outputFile;
	private int tileSizeX;
	private int tileSizeY;

	public TiledOmeTiffConverter(String inputFile, String outputFile, int tileSizeX, int tileSizeY) {
		this.inputFile = inputFile;
		this.outputFile = outputFile;
		this.tileSizeX = tileSizeX;
		this.tileSizeY = tileSizeY;
	}

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
		fileWriters = new ArrayList<Quartet<Integer,Integer,Integer,OMETiffWriter>>();
		for (int z=0; z<reader.getSizeZ(); z++) {
			String outFileZ = outputFile;
			if (reader.getSizeZ()>1) {
				outFileZ = outFileZ.replace(".ome.tif", "_z" + String.format("%03d", z) + ".ome.tif");
			}
			for (int c=0; c<reader.getSizeC(); c++) {
				String outFileZC = outFileZ;
				if (reader.getSizeC()>1) {
					outFileZC = outFileZC.replace(".ome.tif", "_c" + String.format("%03d", c) + ".ome.tif");
				}
				for (int t=0; t<reader.getSizeT(); t++) {
					String outFileZCT = outFileZC;
					if (reader.getSizeT()>1) {
						outFileZCT = outFileZCT.replace(".ome.tif", "_t" + String.format("%03d", t) + ".ome.tif");
					}
					// important to delete because OME uses RandomAccessFile
					Path outputPath = Paths.get(outFileZCT);
					outputPath.toFile().delete();

					// set up the writer and associate it with the output file
					OMETiffWriter writer = new OMETiffWriter();
					writer.setMetadataRetrieve(omexml);
					
				    // set the tile size height and width for writing
				    this.tileSizeX = writer.setTileSizeX(tileSizeX);
				    this.tileSizeY = writer.setTileSizeY(tileSizeY);

					writer.setId(outFileZCT);

					writer.setCompression(CompressionType.LZW.getCompression());
					
					LOG.info("Initializing writer for slice: (z=" + Integer.toString(z) + ",c=" + Integer.toString(c) + ",t=" + Integer.toString(t) + ")");
					
					Quartet<Integer,Integer,Integer,OMETiffWriter> quartet = new Quartet<Integer,Integer,Integer,OMETiffWriter>(z, c, t, writer);	
					fileWriters.add(quartet);
				}
			}
		}
	}

	public void readWriteTiles() throws FormatException, DependencyException, ServiceException, IOException {
		/* This function only saves the first series of a file if multiple series are present. */
		
		int bpp = FormatTools.getBytesPerPixel(reader.getPixelType());
		int tilePlaneSize = tileSizeX * tileSizeY * reader.getRGBChannelCount() * bpp;
		byte[] buf = new byte[tilePlaneSize];

		// convert each image in the current series
		for (int image=0; image<reader.getImageCount(); image++) {
			OMETiffWriter writer = null;
			int[] coords = reader.getZCTCoords(image);
			LOG.info("Writing slice: (z=" + Integer.toString(coords[0]) + ",c=" + Integer.toString(coords[1]) + ",t=" + Integer.toString(coords[2]) + ")");
			for (int w = 0; w<fileWriters.size(); w++) {
				Quartet<Integer, Integer, Integer, OMETiffWriter> quartet = fileWriters.get(w);
				if (quartet.getValue0()==coords[0] &&
					quartet.getValue1()==coords[1] &&
					quartet.getValue2()==coords[2]) {
					writer = quartet.getValue3();
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

			for (int y=0; y<nYTiles; y++) {
				for (int x=0; x<nXTiles; x++) {
					
					int tileX = x * tileSizeX;
					int tileY = y * tileSizeY;
					
					int effTileSizeX = (tileX + tileSizeX) < width ? tileSizeX : width - tileX;
					int effTileSizeY = (tileY + tileSizeY) < height ? tileSizeY : height - tileY;

					buf = reader.openBytes(image, tileX, tileY, effTileSizeX, effTileSizeY);
					writer.saveBytes(image, buf, tileX, tileY, effTileSizeX, effTileSizeY);
				}
			}
		}
	}

	private void cleanup() {
		try {
			reader.close();
		}
		catch (IOException e) {
			LOG.log(Level.WARNING, "Failed to close reader.",e);
		}
		try {
			for (int w=0; w<fileWriters.size(); w++) {
				fileWriters.get(w).getValue3().close();
			}
		}
		catch (IOException e) {
			LOG.log(Level.WARNING, "Failed to close writers.",e);
		}
	}

	// Do the work
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
