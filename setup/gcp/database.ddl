CREATE TABLE GeneratedDatasets (
    DatasetName               STRING(MAX) NOT NULL,
    CreationTime              TIMESTAMP NOT NULL,
    DatasetCreatorVersion     STRING(MAX),
) PRIMARY KEY (DatasetName, CreationTime);

-- The DatasetUnpopulatedExamples table saves examples that only contain
-- metadata. For example, they don't contain frames, but only the metadata of
-- frames (video path, timestamps).
CREATE TABLE DatasetUnpopulatedExamples (
    DatasetName     STRING(MAX) NOT NULL,
    CreationTime    TIMESTAMP NOT NULL,
--     Note that ExampleIndex is the first column after the dataset identifiers,
--     as reading from spanner is lexicographical and we wish to maintain order
--     of the examples when reading them all.
    ExampleIndex    INT64 NOT NULL,
    EncodedExample  BYTES(MAX) NOT NULL,
) PRIMARY KEY (DatasetName, CreationTime, ExampleIndex),
INTERLEAVE IN PARENT GeneratedDatasets ON DELETE CASCADE;