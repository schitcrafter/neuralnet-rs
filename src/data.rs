use std::{error::Error, fs, io::Cursor, path::Path};

use binrw::BinRead;
use log::debug;

pub struct DataSet {
    pub labelled_images: Vec<LabelledImage>,
    /// (number of rows, number of columns)
    pub image_dimensions: (u32, u32),
}

impl DataSet {
    fn from_images_and_labels(images: ImagesFile, labels: LabelsFile) -> DataSet {
        let image_dimensions = (images.num_rows, images.num_columns);

        // Expected number of bytes in one image
        let expected_image_len = image_dimensions.0 * image_dimensions.1;

        assert_eq!(
            images.num_images, labels.num_labels,
            "Same Number of images as labels"
        );

        let labelled_images = images
            .images
            .into_iter()
            .zip(labels.labels)
            .map(|(image, label)| {
                assert_eq!(
                    image.binary.len(),
                    expected_image_len as usize,
                    "Image does not have the expected proportions"
                );
                LabelledImage {
                    image: image.binary,
                    label,
                }
            })
            .collect();

        DataSet {
            labelled_images,
            image_dimensions,
        }
    }
}

pub struct LabelledImage {
    pub image: Vec<u8>,
    pub label: u8,
}

#[derive(BinRead)]
#[br(big, magic = 0x00000803u32)]
struct ImagesFile {
    num_images: u32,
    num_rows: u32,
    num_columns: u32,

    #[br(args {
        count: num_images as usize,
        inner: binrw::args! {
            num_rows,
            num_columns
        }
    })]
    images: Vec<ImageBinary>,
}

#[derive(BinRead)]
#[br(big, import { num_rows: u32, num_columns: u32 })]
struct ImageBinary {
    #[br(count = num_rows * num_columns)]
    binary: Vec<u8>,
}

#[derive(BinRead)]
#[br(big, magic = 0x00000801u32)]
struct LabelsFile {
    num_labels: u32,
    #[br(count = num_labels)]
    labels: Vec<u8>,
}

pub fn read_data(
    images_file: impl AsRef<Path>,
    labels_file: impl AsRef<Path>,
) -> Result<DataSet, Box<dyn Error>> {
    let images_binary = fs::read(images_file)?;
    let labels_binary = fs::read(labels_file)?;

    let images_read = ImagesFile::read(&mut Cursor::new(images_binary))?;

    debug!(
        "Read images file, num_images={} num_rows={} num_columns={}",
        images_read.num_images, images_read.num_rows, images_read.num_columns
    );

    debug!(
        "First image binary data (len={}): {:?}",
        images_read.images[0].binary.len(),
        images_read.images[0].binary
    );

    assert_eq!(images_read.num_images, images_read.images.len() as u32);
    assert_eq!(
        images_read.num_rows * images_read.num_columns,
        images_read.images[0].binary.len() as u32
    );

    let labels_read = LabelsFile::read(&mut Cursor::new(labels_binary))?;

    debug!(
        "Read labels file, number of labels: {}",
        labels_read.num_labels
    );

    let dataset = DataSet::from_images_and_labels(images_read, labels_read);

    Ok(dataset)
}
