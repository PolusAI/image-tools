package axle.polus.data.utils.converters;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.logging.Level;
import java.util.logging.Logger;

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
 * Based off of the example from
 * https://docs.openmicroscopy.org/bio-formats/5.9.1/developers/tiling.html
 * 
 * One optimization this method uses is reading/writing multiple tiles
 * simultaneously (up to 16 at a time). This can be set using the xMulti and
 * yMulti variables.
 *
 * @author nick.schaub at nih.gov
 */
public class TiledOmeTiffConverter implements Runnable {
	/**
	 * Class that handles conversion of any Bioformats supported image format into an OME tiled tiff.
	 */

	private static final Logger LOG = Logger.getLogger(TiledOmeTiffConverter.class.getName());
	
	private ImageReader reader;
	private String inputFile;
	private String outputFile;
	private int tileSizeX = 1024;
	private int tileSizeY = 1024;
	private int xMulti = 4;
	private int yMulti = 4;
	private int Z;
	private int C;
	private int T;

	/**
	 * Class constructor
	 * 
	 * @param reader		ImageReader to convert to .ome.tif
	 * @param outputFile	Complete path to export file
	 * @param Z				The z-slice to export
	 * @param C				The channel slice to export
	 * @param T 			The time-point slice to export
	 */
	public TiledOmeTiffConverter(ImageReader reader, String outputFile, int Z, int C, int T) {
		this.inputFile = reader.getCurrentFile();
		this.outputFile = outputFile;
		this.Z = Z;
		this.C = C;
		this.T = T;
	}
	
	/**
	 * Initialize the OME Tiff writer
	 * 
	 * @param omexml		Base metadata to import
	 * @return
	 * @throws FormatException
	 * @throws IOException
	 */
	public OMETiffWriter init_writer(IMetadata omexml) {

		// important to delete because OME uses RandomAccessFile
		Path outputPath = Paths.get(this.outputFile);
		outputPath.toFile().delete();

		// set up the writer and associate it with the output file
		OMETiffWriter writer = new OMETiffWriter();
		writer.setMetadataRetrieve(omexml);
		
	    // set output file properties
	    try {
			this.tileSizeX = writer.setTileSizeX(tileSizeX);
		    this.tileSizeY = writer.setTileSizeY(tileSizeY);
			writer.setId(this.outputFile);
			writer.setCompression(CompressionType.LZW.getCompression());
		} catch (FormatException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    
		omexml = (IMetadata) writer.getMetadataRetrieve();
		omexml.setPixelsSizeZ(new PositiveInteger(1), 0);
		omexml.setPixelsSizeC(new PositiveInteger(1), 0);
		omexml.setPixelsSizeT(new PositiveInteger(1), 0);
		
		return writer;
	}

	/**
	 * Read input file and write output file in tiles
	 * 
	 * The input files is read in tiles of size (tileSizeX, tileSizeY), with
	 * only one z-slice at a time. Each channel and time-point (if present) are
	 * saved to separate files.
	 * 
	 * It is important to save images in tiles, since Bioformats indexes pixels
	 * using signed integers (32 bits). This means that loading a full image
	 * plane larger than 2GB will throw an indexing error. Saving in tiles also
	 * has the benefit of being memory efficient, so it can be run on small
	 * nodes.
	 * 
	 * @throws FormatException
	 * @throws DependencyException
	 * @throws ServiceException
	 * @throws IOException
	 */
	public void readWriteTiles() throws FormatException, DependencyException, ServiceException, IOException {
		LOG.info("Writing " + (new File(this.outputFile)).getName() + " from " + (new File(this.inputFile)).getName());

		// construct the object that stores OME-XML metadata
		ServiceFactory factory = new ServiceFactory();
		OMEXMLService service = factory.getInstance(OMEXMLService.class);
		IMetadata omexml = service.createOMEXMLMetadata();

		// set up the reader and associate it with the input file
		reader = new ImageReader();
		reader.setOriginalMetadataPopulated(true);
		reader.setMetadataStore(omexml);
		reader.setId(this.inputFile);

		int bpp = FormatTools.getBytesPerPixel(reader.getPixelType());
		int tilePlaneSize = xMulti * yMulti * tileSizeX * tileSizeY * reader.getRGBChannelCount() * bpp;
		byte[] buf = new byte[tilePlaneSize];
					
		OMETiffWriter writer = this.init_writer(omexml);
		
		int width = reader.getSizeX();
		int height = reader.getSizeY();

		// Determined the number of tiles to read and write
		int nXTiles = width / (xMulti * tileSizeX);
		int nYTiles = height / (yMulti * tileSizeY);
		if (nXTiles * tileSizeX * xMulti != width) nXTiles++;
		if (nYTiles * tileSizeY * yMulti != height) nYTiles++;
	
		int index = reader.getIndex(this.Z, this.C, this.T);
		for (int y=0; y<nYTiles; y++) {
			for (int x=0; x<nXTiles; x++) {
				
				int tileX = x * tileSizeX * xMulti;
				int tileY = y * tileSizeY * yMulti;

				
				int effTileSizeX = (tileX + tileSizeX * xMulti) < width ? tileSizeX * xMulti: width - tileX;
				int effTileSizeY = (tileY + tileSizeY * yMulti) < height ? tileSizeY * yMulti : height - tileY;

				buf = reader.openBytes(index, tileX, tileY, effTileSizeX, effTileSizeY);
				writer.saveBytes(0, buf, tileX, tileY, effTileSizeX, effTileSizeY);
			}
		}

		// Close the readers/writers
		try {
			writer.close();
		}
		catch (IOException e) {
			LOG.log(Level.WARNING, reader.toString() + ": Failed to close writer.",e);
		}
		try {
			reader.close();
		}
		catch (IOException e) {
			LOG.log(Level.WARNING, reader.toString() + ": Failed to close reader.",e);
		}
	}

	/**
	 * Main process, set up for threading
	 */
	public void run() {

		try {
			// read and write the image using tiles
			this.readWriteTiles();
		}
		catch(Exception e) {
			System.err.println("Failed to read and write tiles for image: " + new File(this.outputFile).getName());
			e.printStackTrace();
		}
	}

}
