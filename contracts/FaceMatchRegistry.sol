// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FaceMatchRegistry
 * @dev Production-grade tamper-evident ledger for recording and independently
 *      verifying multi-site biometric identity consensus records.
 */
contract FaceMatchRegistry {
    error RecordNotFound(uint256 id);
    error ConsensusNotFound(uint256 id);
    error InvalidInput();

    // --- Single-Match Record (Legacy & Simple Match Support) ---
    struct MatchRecord {
        bytes32 faceHash;         // SHA-256 hash of ArcFace 512-d embedding vector
        bytes32 matchHash;        // Cryptographic digest of discovered match payload
        string matchUrl;          // Discovered social media post or profile URL
        string platform;          // e.g. "Instagram", "Twitter/X", "LinkedIn", "Reddit", "Web"
        string metadataJson;      // Forensic metadata (title, snippet, detector, model, timestamp)
        uint256 timestamp;        // Block timestamp when registered
        address recordedBy;       // Submitting address
    }

    // --- Multi-Site Identity Consensus Record ---
    struct ConsensusRecord {
        bytes32 faceHash;           // SHA-256 hash of ArcFace 512-d embedding vector
        bytes32 merkleRoot;         // Merkle root combining face, site URLs, and avatar digests
        string entityName;          // Finalized consensus entity/person name
        string[] platforms;         // List of verified platforms (e.g. ["Twitter/X", "LinkedIn", "GitHub"])
        string[] matchUrls;         // Discovered matching URLs across all sites
        uint256 consensusScore;     // Scaled 0 - 10000 (e.g. 9840 = 98.40%)
        uint256 verifiedSiteCount;  // Count of independent platforms in agreement
        string metadataJson;        // Full forensic audit payload (per-site similarity scores, handles)
        uint256 timestamp;          // Block timestamp
        address recordedBy;         // Submitting wallet address
    }

    mapping(uint256 => MatchRecord) private _records;
    uint256 public recordCount;

    mapping(uint256 => ConsensusRecord) private _consensusRecords;
    uint256 public consensusCount;

    // Events
    event FaceMatchRecorded(
        uint256 indexed recordId,
        bytes32 indexed faceHash,
        bytes32 indexed matchHash,
        string matchUrl,
        string platform,
        uint256 timestamp,
        address recordedBy
    );

    event ConsensusMatchRecorded(
        uint256 indexed consensusId,
        bytes32 indexed faceHash,
        bytes32 indexed merkleRoot,
        string entityName,
        uint256 consensusScore,
        uint256 verifiedSiteCount,
        uint256 timestamp,
        address recordedBy
    );

    // ==========================================
    // MULTI-SITE CONSENSUS LEDGER FUNCTIONS
    // ==========================================

    /**
     * @notice Records a verified cross-platform identity consensus into the blockchain
     */
    function recordConsensusMatch(
        bytes32 _faceHash,
        bytes32 _merkleRoot,
        string calldata _entityName,
        string[] calldata _platforms,
        string[] calldata _matchUrls,
        uint256 _consensusScore,
        uint256 _verifiedSiteCount,
        string calldata _metadataJson
    ) external returns (uint256) {
        if (_verifiedSiteCount == 0 || _platforms.length != _matchUrls.length) {
            revert InvalidInput();
        }

        uint256 id = consensusCount++;
        _consensusRecords[id] = ConsensusRecord({
            faceHash: _faceHash,
            merkleRoot: _merkleRoot,
            entityName: _entityName,
            platforms: _platforms,
            matchUrls: _matchUrls,
            consensusScore: _consensusScore,
            verifiedSiteCount: _verifiedSiteCount,
            metadataJson: _metadataJson,
            timestamp: block.timestamp,
            recordedBy: msg.sender
        });

        emit ConsensusMatchRecorded(
            id,
            _faceHash,
            _merkleRoot,
            _entityName,
            _consensusScore,
            _verifiedSiteCount,
            block.timestamp,
            msg.sender
        );

        return id;
    }

    /**
     * @notice Retrieves a multi-site consensus record by ID
     */
    function getConsensusRecord(uint256 _id) external view returns (ConsensusRecord memory) {
        if (_id >= consensusCount) {
            revert ConsensusNotFound(_id);
        }
        return _consensusRecords[_id];
    }

    /**
     * @notice Verifies whether a supplied face hash and Merkle root match the immutable record
     */
    function verifyConsensusRecord(
        uint256 _id,
        bytes32 _expectedFaceHash,
        bytes32 _expectedMerkleRoot
    ) external view returns (bool isVerified, uint256 consensusScore, uint256 siteCount, string memory entityName) {
        if (_id >= consensusCount) {
            revert ConsensusNotFound(_id);
        }
        ConsensusRecord storage rec = _consensusRecords[_id];
        bool valid = (rec.faceHash == _expectedFaceHash && rec.merkleRoot == _expectedMerkleRoot);
        return (valid, rec.consensusScore, rec.verifiedSiteCount, rec.entityName);
    }

    // ==========================================
    // SINGLE-MATCH FUNCTIONS (BACKWARD COMPAT)
    // ==========================================

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

    function getRecord(uint256 _id) external view returns (MatchRecord memory) {
        if (_id >= recordCount) {
            revert RecordNotFound(_id);
        }
        return _records[_id];
    }

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
