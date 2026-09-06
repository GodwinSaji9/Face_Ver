from dataclasses import dataclass, asdict
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from web3 import Web3
from web3.exceptions import Web3Exception
from src.utils import to_bytes32, format_bytes32_hex


# Standard Anvil default accounts
ANVIL_DEFAULT_ACCOUNT = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
ANVIL_DEFAULT_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
CONTRACT_CACHE_FILE = ".last_contract"


@dataclass
class BlockchainRecordResult:
    record_id: int
    contract_address: str
    tx_hash: str
    block_number: int
    gas_used: int
    face_hash: str
    match_hash: str
    match_url: str
    platform: str
    recorded_by: str
    timestamp: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BlockchainConsensusResult:
    consensus_id: int
    contract_address: str
    tx_hash: str
    block_number: int
    gas_used: int
    face_hash: str
    merkle_root: str
    entity_name: str
    consensus_score: float
    verified_site_count: int
    platforms: List[str]
    match_urls: List[str]
    recorded_by: str
    timestamp: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OnChainVerificationResult:
    record_id: int
    is_verified: bool
    face_hash_matches: bool
    match_hash_matches: bool
    on_chain_face_hash: str
    on_chain_match_hash: str
    on_chain_url: str
    on_chain_platform: str
    on_chain_timestamp: int
    recorded_by: str
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OnChainConsensusVerificationResult:
    consensus_id: int
    is_verified: bool
    face_hash_matches: bool
    merkle_root_matches: bool
    on_chain_face_hash: str
    on_chain_merkle_root: str
    entity_name: str
    consensus_score: float
    verified_site_count: int
    platforms: List[str]
    match_urls: List[str]
    on_chain_timestamp: int
    recorded_by: str
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


class BlockchainEngine:
    """
    Interfaces with Foundry Anvil Ethereum node and the FaceMatchRegistry smart contract.
    Handles automated contract deployment, match recording, and tamper-evident verification.
    """

    def __init__(
        self,
        rpc_url: str = "http://127.0.0.1:8545",
        private_key: Optional[str] = None,
        contract_address: Optional[str] = None,
    ):
        self.rpc_url = rpc_url
        self.private_key = private_key or os.getenv("ANVIL_PRIVATE_KEY") or ANVIL_DEFAULT_PRIVATE_KEY
        
        # Determine contract address (param -> env -> cache file)
        if contract_address:
            self.contract_address = contract_address
        elif os.getenv("CONTRACT_ADDRESS"):
            self.contract_address = os.getenv("CONTRACT_ADDRESS")
        elif os.path.exists(CONTRACT_CACHE_FILE):
            try:
                with open(CONTRACT_CACHE_FILE, "r") as f:
                    self.contract_address = f.read().strip()
            except Exception:
                self.contract_address = None
        else:
            self.contract_address = None

        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        self.account = self.w3.eth.account.from_key(self.private_key)
        self.abi, self.bytecode = self._load_artifact()

    def _load_artifact(self) -> Tuple[list, str]:
        """Loads compiled ABI and bytecode from local artifact file."""
        artifact_path = os.path.join(os.path.dirname(__file__), "contract_artifact.json")
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Smart contract artifact not found at {artifact_path}")
        with open(artifact_path, "r") as f:
            data = json.load(f)
        return data["abi"], data["bytecode"]

    def is_connected(self) -> bool:
        """Returns True if connected to Anvil RPC node."""
        try:
            return self.w3.is_connected()
        except Exception:
            return False

    @staticmethod
    def launch_anvil_daemon() -> subprocess.Popen:
        """
        Launches Foundry Anvil in the background as a detached daemon with state persistence.
        """
        anvil_cmd = shutil.which("anvil")
        if not anvil_cmd:
            foundry_bin = os.path.expanduser("~/.foundry/bin/anvil.exe")
            if os.path.exists(foundry_bin):
                anvil_cmd = foundry_bin
            else:
                raise RuntimeError(
                    "Anvil binary not found. Please install Foundry or ensure anvil is in PATH."
                )

        state_file = os.path.abspath("anvil_state.json")
        cmd = [
            anvil_cmd,
            "--port", "8545",
            "--state", state_file,
            "--state-interval", "2",
            "--silent",
        ]

        creation_flags = 0
        if sys.platform == "win32":
            # DETACHED_PROCESS (0x08) | CREATE_NEW_PROCESS_GROUP (0x200)
            creation_flags = 0x00000008 | 0x00000200

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        # Wait briefly for RPC server to bind
        time.sleep(1.5)
        return proc

    def ensure_connected(self, auto_start: bool = True) -> None:
        """Ensures active RPC connection, optionally starting Anvil automatically."""
        if not self.is_connected():
            if auto_start:
                self.launch_anvil_daemon()
                # Re-check connection
                for _ in range(15):
                    if self.is_connected():
                        return
                    time.sleep(0.5)
            raise ConnectionError(
                f"Could not connect to Ethereum node at {self.rpc_url}. "
                "Ensure Anvil is running ('anvil' in terminal)."
            )

    def is_contract_deployed(self, address: Optional[str] = None) -> bool:
        """Checks if byte code exists at the given contract address on the connected chain."""
        target = address or self.contract_address
        if not target:
            return False
        try:
            checksummed = self.w3.to_checksum_address(target)
            code = self.w3.eth.get_code(checksummed)
            return len(code) > 2
        except Exception:
            return False

    def _get_resilient_tx_params(self, gas_limit: int = 800_000) -> Dict[str, Any]:
        """
        Dynamically estimates gas price with safety buffer and uses pending nonce (Iteration 15).
        Prevents transaction underpricing and nonce collision.
        """
        nonce = self.w3.eth.get_transaction_count(self.account.address, "pending")
        base_gas_price = self.w3.eth.gas_price
        buffered_gas_price = int(base_gas_price * 1.15) if base_gas_price > 0 else 1_000_000_000

        return {
            "from": self.account.address,
            "nonce": nonce,
            "gas": gas_limit,
            "gasPrice": buffered_gas_price,
            "chainId": self.w3.eth.chain_id,
        }

    def deploy_contract(self) -> str:
        """
        Deploys FaceMatchRegistry.sol to the connected Anvil node.
        Returns deployed contract address and caches it.
        """
        self.ensure_connected()
        contract_factory = self.w3.eth.contract(abi=self.abi, bytecode=self.bytecode)
        
        tx = contract_factory.constructor().build_transaction(
            self._get_resilient_tx_params(gas_limit=2_000_000)
        )

        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        if receipt.status != 1:
            raise RuntimeError(f"Contract deployment failed! Tx: {tx_hash.hex()}")

        self.contract_address = receipt.contractAddress
        
        # Save to cache file for subsequent runs
        try:
            with open(CONTRACT_CACHE_FILE, "w") as f:
                f.write(self.contract_address)
        except Exception:
            pass

        return self.contract_address

    def get_contract(self):
        """Returns instantiated Web3 contract instance, auto-deploying if not present on chain."""
        if not self.contract_address or not self.is_contract_deployed(self.contract_address):
            self.deploy_contract()
        return self.w3.eth.contract(address=self.w3.to_checksum_address(self.contract_address), abi=self.abi)

    def record_match(
        self,
        face_hash: str,
        match_hash: str,
        match_url: str,
        platform: str,
        metadata: Dict[str, Any],
    ) -> BlockchainRecordResult:
        """
        Submits a face match record to the blockchain registry.
        """
        self.ensure_connected()
        contract = self.get_contract()

        face_bytes = to_bytes32(face_hash)
        match_bytes = to_bytes32(match_hash)
        metadata_json = json.dumps(metadata, sort_keys=True)

        try:
            est_gas = contract.functions.recordMatch(
                face_bytes,
                match_bytes,
                match_url,
                platform,
                metadata_json,
            ).estimate_gas({"from": self.account.address})
            gas_limit = int(est_gas * 1.30)
        except Exception:
            gas_limit = 1_000_000

        tx = contract.functions.recordMatch(
            face_bytes,
            match_bytes,
            match_url,
            platform,
            metadata_json,
        ).build_transaction(self._get_resilient_tx_params(gas_limit=gas_limit))

        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        if receipt.status != 1:
            raise RuntimeError(f"recordMatch transaction failed: {tx_hash.hex()}")

        # Extract recordId from event logs
        processed_logs = contract.events.FaceMatchRecorded().process_receipt(receipt)
        if processed_logs:
            record_id = processed_logs[0]["args"]["recordId"]
            block_timestamp = processed_logs[0]["args"]["timestamp"]
        else:
            # Fallback: total records minus 1
            record_id = contract.functions.recordCount().call() - 1
            block = self.w3.eth.get_block(receipt.blockNumber)
            block_timestamp = block["timestamp"]

        return BlockchainRecordResult(
            record_id=record_id,
            contract_address=self.contract_address,
            tx_hash="0x" + tx_hash.hex(),
            block_number=receipt.blockNumber,
            gas_used=receipt.gasUsed,
            face_hash=face_hash,
            match_hash=match_hash,
            match_url=match_url,
            platform=platform,
            recorded_by=self.account.address,
            timestamp=block_timestamp,
        )

    def verify_record(
        self,
        record_id: int,
        expected_face_hash: str,
        expected_match_hash: str,
    ) -> OnChainVerificationResult:
        """
        Re-queries the blockchain to verify recorded data and validate tamper-evidence.
        """
        self.ensure_connected()
        contract = self.get_contract()

        raw_record = contract.functions.getRecord(record_id).call()
        # raw_record: (faceHash, matchHash, matchUrl, platform, metadataJson, timestamp, recordedBy)
        on_chain_face_bytes = raw_record[0]
        on_chain_match_bytes = raw_record[1]
        on_chain_url = raw_record[2]
        on_chain_platform = raw_record[3]
        on_chain_metadata_raw = raw_record[4]
        on_chain_timestamp = raw_record[5]
        on_chain_recorder = raw_record[6]

        expected_face_bytes = to_bytes32(expected_face_hash)
        expected_match_bytes = to_bytes32(expected_match_hash)

        face_matches = (on_chain_face_bytes == expected_face_bytes)
        match_matches = (on_chain_match_bytes == expected_match_bytes)

        # Also call smart contract's internal verifyRecord view function
        contract_verified = contract.functions.verifyRecord(
            record_id,
            expected_face_bytes,
            expected_match_bytes,
        ).call()

        try:
            metadata_dict = json.loads(on_chain_metadata_raw)
        except Exception:
            metadata_dict = {"raw": on_chain_metadata_raw}

        return OnChainVerificationResult(
            record_id=record_id,
            is_verified=(face_matches and match_matches and contract_verified),
            face_hash_matches=face_matches,
            match_hash_matches=match_matches,
            on_chain_face_hash=on_chain_face_bytes.hex(),
            on_chain_match_hash=on_chain_match_bytes.hex(),
            on_chain_url=on_chain_url,
            on_chain_platform=on_chain_platform,
            on_chain_timestamp=on_chain_timestamp,
            recorded_by=on_chain_recorder,
            metadata=metadata_dict,
        )

    def record_consensus_match(
        self,
        face_hash: str,
        merkle_root: str,
        entity_name: str,
        platforms: List[str],
        match_urls: List[str],
        consensus_score: float,
        verified_site_count: int,
        metadata: Dict[str, Any],
    ) -> BlockchainConsensusResult:
        """
        Submits a multi-site verified identity consensus record to the smart contract.
        """
        self.ensure_connected()
        contract = self.get_contract()

        face_bytes = to_bytes32(face_hash)
        merkle_bytes = to_bytes32(merkle_root)
        score_scaled = int(round(consensus_score * 10000))
        metadata_json = json.dumps(metadata, sort_keys=True)

        try:
            est_gas = contract.functions.recordConsensusMatch(
                face_bytes,
                merkle_bytes,
                entity_name,
                platforms,
                match_urls,
                score_scaled,
                verified_site_count,
                metadata_json,
            ).estimate_gas({"from": self.account.address})
            gas_limit = int(est_gas * 1.30)
        except Exception:
            gas_limit = 2_000_000

        tx = contract.functions.recordConsensusMatch(
            face_bytes,
            merkle_bytes,
            entity_name,
            platforms,
            match_urls,
            score_scaled,
            verified_site_count,
            metadata_json,
        ).build_transaction(self._get_resilient_tx_params(gas_limit=gas_limit))

        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        if receipt.status != 1:
            raise RuntimeError(f"recordConsensusMatch transaction failed: {tx_hash.hex()}")

        processed_logs = contract.events.ConsensusMatchRecorded().process_receipt(receipt)
        if processed_logs:
            consensus_id = processed_logs[0]["args"]["consensusId"]
            block_timestamp = processed_logs[0]["args"]["timestamp"]
        else:
            consensus_id = contract.functions.consensusCount().call() - 1
            block = self.w3.eth.get_block(receipt.blockNumber)
            block_timestamp = block["timestamp"]

        return BlockchainConsensusResult(
            consensus_id=consensus_id,
            contract_address=self.contract_address,
            tx_hash="0x" + tx_hash.hex(),
            block_number=receipt.blockNumber,
            gas_used=receipt.gasUsed,
            face_hash=face_hash,
            merkle_root=merkle_root,
            entity_name=entity_name,
            consensus_score=consensus_score,
            verified_site_count=verified_site_count,
            platforms=platforms,
            match_urls=match_urls,
            recorded_by=self.account.address,
            timestamp=block_timestamp,
        )

    def verify_consensus_record(
        self,
        consensus_id: int,
        expected_face_hash: str,
        expected_merkle_root: str,
    ) -> OnChainConsensusVerificationResult:
        """
        Re-queries the blockchain to verify a multi-site consensus record for tamper-evidence.
        """
        self.ensure_connected()
        contract = self.get_contract()

        raw_record = contract.functions.getConsensusRecord(consensus_id).call()
        on_chain_face_bytes = raw_record[0]
        on_chain_merkle_bytes = raw_record[1]
        on_chain_entity = raw_record[2]
        on_chain_platforms = list(raw_record[3])
        on_chain_urls = list(raw_record[4])
        on_chain_score_scaled = raw_record[5]
        on_chain_verified_count = raw_record[6]
        on_chain_meta_raw = raw_record[7]
        on_chain_ts = raw_record[8]
        on_chain_recorder = raw_record[9]

        expected_face_bytes = to_bytes32(expected_face_hash)
        expected_merkle_bytes = to_bytes32(expected_merkle_root)

        face_matches = (on_chain_face_bytes == expected_face_bytes)
        merkle_matches = (on_chain_merkle_bytes == expected_merkle_bytes)

        verify_res = contract.functions.verifyConsensusRecord(
            consensus_id,
            expected_face_bytes,
            expected_merkle_bytes,
        ).call()
        contract_verified = verify_res[0]

        try:
            meta_dict = json.loads(on_chain_meta_raw)
        except Exception:
            meta_dict = {"raw": on_chain_meta_raw}

        return OnChainConsensusVerificationResult(
            consensus_id=consensus_id,
            is_verified=(face_matches and merkle_matches and contract_verified),
            face_hash_matches=face_matches,
            merkle_root_matches=merkle_matches,
            on_chain_face_hash=on_chain_face_bytes.hex(),
            on_chain_merkle_root=on_chain_merkle_bytes.hex(),
            entity_name=on_chain_entity,
            consensus_score=float(on_chain_score_scaled / 10000.0),
            verified_site_count=on_chain_verified_count,
            platforms=on_chain_platforms,
            match_urls=on_chain_urls,
            on_chain_timestamp=on_chain_ts,
            recorded_by=on_chain_recorder,
            metadata=meta_dict,
        )

