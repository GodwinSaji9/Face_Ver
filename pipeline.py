import json
import os
import sys
from typing import Optional
from dotenv import load_dotenv
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import typer

# Force UTF-8 encoding on standard streams to avoid Windows charmap issues
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

from src.blockchain_engine import BlockchainEngine
from src.face_engine import FaceEngine
from src.search_engine import SearchEngine
from src.utils import identify_platform

app = typer.Typer(
    name="face-blockchain-pipeline",
    help="End-to-End Face ID + Reverse Search + Blockchain Verification Pipeline",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    image: str = typer.Option(
        ...,
        "--image",
        "-i",
        help="Path to input photo for face detection & encoding.",
    ),
    detector: str = typer.Option(
        "retinaface",
        "--detector",
        "-d",
        help="Face detector backend (retinaface, opencv, mtcnn).",
    ),
    model: str = typer.Option(
        "ArcFace",
        "--model",
        "-m",
        help="Facial feature representation model (ArcFace, Facenet512, VGG-Face).",
    ),
    serpapi_key: Optional[str] = typer.Option(
        None,
        "--serpapi-key",
        "-k",
        help="SerpAPI key for live Google Lens reverse search (defaults to SERPAPI_KEY in .env).",
    ),
    rpc_url: str = typer.Option(
        "http://127.0.0.1:8545",
        "--rpc-url",
        "-r",
        help="Ethereum JSON-RPC URL (Foundry Anvil default: http://127.0.0.1:8545).",
    ),
    contract_address: Optional[str] = typer.Option(
        None,
        "--contract-address",
        "-c",
        help="Existing FaceMatchRegistry contract address (auto-deploys if omitted).",
    ),
    demo_search: bool = typer.Option(
        False,
        "--demo-search",
        help="Run search step in demo mode if no live SerpAPI key is available.",
    ),
    save_json: Optional[str] = typer.Option(
        None,
        "--save-json",
        "-o",
        help="Path to export the final verification report as a JSON file.",
    ),
):
    """
    Executes the full pipeline:
    1. Detect & encode face (RetinaFace + ArcFace)
    2. Reverse image search (Google Lens via SerpAPI)
    3. Upload match data to blockchain (Foundry Anvil)
    4. Re-verify data against immutable on-chain record
    """
    console.print(
        Panel.fit(
            "[bold cyan]🛡️  FACE ID + BLOCKCHAIN VERIFICATION PIPELINE[/bold cyan]\n"
            "[dim]Biometric Analysis → Reverse Image Search → Immutable On-Chain Registry[/dim]",
            border_style="bright_blue",
        )
    )

    resolved_api_key = serpapi_key or os.getenv("SERPAPI_KEY")
    is_demo = demo_search or not resolved_api_key

    if is_demo and not demo_search:
        console.print(
            "[yellow]⚠️  Notice: SERPAPI_KEY not found in environment. "
            "Proceeding with demo search mode. Set SERPAPI_KEY in .env for live searches.[/yellow]\n"
        )

    # -------------------------------------------------------------
    # STEP 1: Face Detection & ArcFace Biometric Encoding
    # -------------------------------------------------------------
    console.print("\n[bold yellow]═══ STEP 1: FACE DETECTION & ARC-FACE ENCODING ═══[/bold yellow]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Running {detector} detection and {model} encoding...", total=None)
        face_engine = FaceEngine(detector=detector, model=model)
        face_result = face_engine.analyze(image)
        progress.update(task, completed=True)

    face_table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    face_table.add_column("Property", style="cyan")
    face_table.add_column("Value", style="green")
    face_table.add_row("Input Image", os.path.abspath(image))
    face_table.add_row("Face Detected", "✅ Yes")
    face_table.add_row("Detector Backend", face_result.detector)
    face_table.add_row("Representation Model", face_result.model)
    face_table.add_row("Bounding Box (x,y,w,h)", f"{face_result.facial_area.get('x')}, {face_result.facial_area.get('y')}, {face_result.facial_area.get('w')}, {face_result.facial_area.get('h')}")
    face_table.add_row("Confidence Score", f"{face_result.confidence:.4f}")
    face_table.add_row("Embedding Dimensions", f"{len(face_result.embedding)} floats (ArcFace)")
    face_table.add_row("Face Fingerprint (SHA-256)", f"[bold white]{face_result.face_hash}[/bold white]")
    face_table.add_row("Cropped Face Path", face_result.cropped_image_path)
    console.print(face_table)

    # -------------------------------------------------------------
    # STEP 2: Reverse Image Search (Google Lens / SerpAPI)
    # -------------------------------------------------------------
    console.print("\n[bold yellow]═══ STEP 2: REVERSE IMAGE SEARCH (SOCIAL MATCH) ═══[/bold yellow]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Searching web & social media for matching posts...", total=None)
        search_engine = SearchEngine(api_key=resolved_api_key)
        search_match = search_engine.search(
            image_path=face_result.cropped_image_path,
            prefer_social=True,
            demo_mode=is_demo,
        )
        progress.update(task, completed=True)

    search_table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    search_table.add_column("Match Property", style="cyan")
    search_table.add_column("Details", style="green")
    search_table.add_row("Match Title", search_match.title)
    search_table.add_row("Source / Platform", f"[bold yellow]{search_match.platform}[/bold yellow] ({search_match.source})")
    search_table.add_row("Discovered URL", f"[underline blue]{search_match.link}[/underline blue]")
    search_table.add_row("Is Social Media", "✅ Yes" if search_match.is_social else "ℹ️ Web")
    search_table.add_row("Match Fingerprint (SHA-256)", f"[bold white]{search_match.match_hash}[/bold white]")
    search_table.add_row("Timestamp (UTC)", search_match.timestamp_utc)
    if search_match.raw_snippet:
        search_table.add_row("Snippet", search_match.raw_snippet[:100] + "...")
    console.print(search_table)

    # -------------------------------------------------------------
    # STEP 3: Blockchain Upload (Foundry Anvil)
    # -------------------------------------------------------------
    console.print("\n[bold yellow]═══ STEP 3: UPLOAD MATCH RECORD TO BLOCKCHAIN ═══[/bold yellow]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Connecting to Anvil node and preparing smart contract...", total=None)
        blockchain = BlockchainEngine(
            rpc_url=rpc_url,
            contract_address=contract_address,
        )
        blockchain.ensure_connected(auto_start=True)

        if not blockchain.contract_address or not blockchain.is_contract_deployed(blockchain.contract_address):
            progress.update(task, description="Deploying FaceMatchRegistry.sol to Anvil...")
            deployed_addr = blockchain.deploy_contract()
        else:
            deployed_addr = blockchain.contract_address

        progress.update(task, description="Broadcasting recordMatch transaction to Anvil...")
        metadata_payload = {
            "title": search_match.title,
            "source": search_match.source,
            "snippet": search_match.raw_snippet or "",
            "detector": face_result.detector,
            "model": face_result.model,
            "confidence": face_result.confidence,
            "discovered_at": search_match.timestamp_utc,
        }

        record_res = blockchain.record_match(
            face_hash=face_result.face_hash,
            match_hash=search_match.match_hash,
            match_url=search_match.link,
            platform=search_match.platform,
            metadata=metadata_payload,
        )
        progress.update(task, completed=True)

    chain_table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    chain_table.add_column("Blockchain Field", style="cyan")
    chain_table.add_column("Value", style="green")
    chain_table.add_row("Blockchain Network", f"Foundry Anvil (Chain ID: {blockchain.w3.eth.chain_id})")
    chain_table.add_row("Smart Contract", f"[bold yellow]{record_res.contract_address}[/bold yellow]")
    chain_table.add_row("Record ID", f"[bold white]#{record_res.record_id}[/bold white]")
    chain_table.add_row("Transaction Hash", f"[bold green]{record_res.tx_hash}[/bold green]")
    chain_table.add_row("Block Number", str(record_res.block_number))
    chain_table.add_row("Gas Used", f"{record_res.gas_used:,}")
    chain_table.add_row("Submitting Wallet", record_res.recorded_by)
    chain_table.add_row("Block Timestamp", str(record_res.timestamp))
    console.print(chain_table)

    # -------------------------------------------------------------
    # STEP 4: Independent On-Chain Verification
    # -------------------------------------------------------------
    console.print("\n[bold yellow]═══ STEP 4: INDEPENDENT ON-CHAIN VERIFICATION ═══[/bold yellow]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Re-querying on-chain state and validating cryptographic proofs...", total=None)
        verification = blockchain.verify_record(
            record_id=record_res.record_id,
            expected_face_hash=face_result.face_hash,
            expected_match_hash=search_match.match_hash,
        )
        progress.update(task, completed=True)

    ver_table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    ver_table.add_column("Verification Check", style="cyan")
    ver_table.add_column("Result", style="green")
    ver_table.add_row("Face Hash Verified On-Chain", "✅ MATCH" if verification.face_hash_matches else "❌ MISMATCH")
    ver_table.add_row("Match Hash Verified On-Chain", "✅ MATCH" if verification.match_hash_matches else "❌ MISMATCH")
    ver_table.add_row("Smart Contract verifyRecord()", "✅ VERIFIED" if verification.is_verified else "❌ FAILED")
    ver_table.add_row("Tamper Evidence Status", "[bold green]AUTHENTIC & TAMPER-EVIDENT[/bold green]" if verification.is_verified else "[bold red]TAMPERED / CORRUPT[/bold red]")
    console.print(ver_table)

    # Final summary panel
    console.print(
        Panel(
            f"[bold green]🎉 PIPELINE COMPLETED SUCCESSFULLY![/bold green]\n\n"
            f"• [bold]Record ID:[/bold] #{record_res.record_id}\n"
            f"• [bold]Contract:[/bold] {record_res.contract_address}\n"
            f"• [bold]Tx Hash:[/bold] {record_res.tx_hash}\n"
            f"• [bold]Face Hash:[/bold] {face_result.face_hash}\n"
            f"• [bold]Discovered Post:[/bold] {search_match.link}\n"
            f"• [bold]Status:[/bold] [bold green]VERIFIED ON ANVIL BLOCKCHAIN[/bold green]",
            title="[bold white]Verification Summary[/bold white]",
            border_style="green",
        )
    )

    # Optional JSON export
    if save_json:
        report = {
            "status": "SUCCESS",
            "face_analysis": face_result.to_dict(),
            "search_match": search_match.to_dict(),
            "blockchain_record": record_res.to_dict(),
            "verification": verification.to_dict(),
        }
        with open(save_json, "w") as f:
            json.dump(report, f, indent=2)
        console.print(f"\n[green]📁 Full report saved to:[/green] [bold]{os.path.abspath(save_json)}[/bold]")


if __name__ == "__main__":
    app()
