// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FaceMatchRegistry
 * @dev Tamper-evident ledger for recording and verifying facial recognition matches
 *      linked to real-world social media and web discoveries.
 */
contract FaceMatchRegistry {
    error RecordNotFound(uint256 id);
    error InvalidInput();

    struct MatchRecord {
        bytes32 faceHash;         // SHA-256 / Keccak-256 hash of ArcFace 512-d embedding vector
        bytes32 matchHash;        // Cryptographic digest of discovered match payload
        string matchUrl;          // Discovered social media post or profile URL
        string platform;          // e.g. "Instagram", "Twitter/X", "LinkedIn", "Reddit", "Web"
        string metadataJson;      // Forensic metadata (title, snippet, detector, model, timestamp)
        uint256 timestamp;        // Block timestamp when registered
        address recordedBy;       // Submitting address
    }

    // Mapping from record ID to MatchRecord
    mapping(uint256 => MatchRecord) private _records;
    uint256 public recordCount;

    // Emitted when a new face match is permanently recorded
    event FaceMatchRecorded(
        uint256 indexed recordId,
        bytes32 indexed faceHash,
        bytes32 indexed matchHash,
        string matchUrl,
        string platform,
        uint256 timestamp,
        address recordedBy
    );

    /**
     * @notice Records a verified facial match into the blockchain
     * @param _faceHash Cryptographic hash of the face embedding
     * @param _matchHash Cryptographic hash of the match data
     * @param _matchUrl Discovered social media URL
     * @param _platform Identified platform/domain
     * @param _metadataJson JSON string with forensic details
     * @return recordId The unique index of the created record
     */
    function recordMatch(
        bytes32 _faceHash,
        bytes32 _matchHash,
        string calldata _matchUrl,
        string calldata _platform,
        string calldata _metadataJson
    ) external returns (uint256) {
        uint256 id = recordCount++;
        _records[id] = MatchRecord({
            faceHash: _faceHash,
            matchHash: _matchHash,
            matchUrl: _matchUrl,
            platform: _platform,
            metadataJson: _metadataJson,
            timestamp: block.timestamp,
            recordedBy: msg.sender
        });

        emit FaceMatchRecorded(
            id,
            _faceHash,
            _matchHash,
            _matchUrl,
            _platform,
            block.timestamp,
            msg.sender
        );

        return id;
    }

    /**
     * @notice Retrieves a record by ID
     * @param _id The record identifier
     * @return The MatchRecord struct
     */
    function getRecord(uint256 _id) external view returns (MatchRecord memory) {
        if (_id >= recordCount) {
            revert RecordNotFound(_id);
        }
        return _records[_id];
    }

    /**
     * @notice Verifies whether a supplied face hash and match hash match the recorded values
     * @param _id The record identifier
     * @param _expectedFaceHash Expected hash of face embedding
     * @param _expectedMatchHash Expected hash of match metadata
     * @return isVerified True if both hashes match the immutable on-chain record
     */
    function verifyRecord(
        uint256 _id,
        bytes32 _expectedFaceHash,
        bytes32 _expectedMatchHash
    ) external view returns (bool isVerified) {
        if (_id >= recordCount) {
            revert RecordNotFound(_id);
        }
        MatchRecord storage rec = _records[_id];
        return (rec.faceHash == _expectedFaceHash && rec.matchHash == _expectedMatchHash);
    }
}
