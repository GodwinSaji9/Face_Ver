import json
import os
import sys
from typing import List, Optional
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
from src.consensus_engine import IdentityConsensusEngine, IdentityConsensusResult
from src.face_engine import FaceEngine
from src.search_engine import SearchEngine
from src.utils import cleanup_temporary_files, identify_platform, select_image_file

app = typer.Typer(
    name="face-blockchain-pipeline",
    help="End-to-End Face ID + Multi-Site Identity Consensus + Blockchain Verification Pipeline",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    image: Optional[str] = typer.Option(
        None,
        "--image",
        "-i",
        help="Path to input photo for face detection & encoding (opens system file dialog if omitted).",
    ),
    choose_file: bool = typer.Option(
        False,
        "--choose-file",
        "--browse",
        "-b",
        help="Open native system file chooser dialog to pick an image from your system.",
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
    threshold: float = typer.Option(
        0.60,
        "--threshold",
        "-t",
        help="Biometric cosine similarity threshold for ArcFace (default: 0.60, recommended 0.55-0.65).",
    ),
    cleanup: bool = typer.Option(
        False,
        "--cleanup",
        help="Clean up temporary face crops and downloaded working files upon completion (default: False, preserves files).",
    ),
    serpapi_key: Optional[str] = typer.Option(
        None,
        "--serpapi-key",
        "-k",
        help="SerpAPI key for live Google Lens reverse search (defaults to SERPAPI_KEY in .env).",
    ),
    serper_api_key: Optional[str] = typer.Option(
        None,
        "--serper-api-key",
        help="Serper.dev API key for Google Search across platforms (defaults to SERPER_API_KEY in .env).",
    ),
    entity_name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Target entity or person name hint (e.g. --name 'Allu Arjun') for live search.",
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
    simulate_disparity: bool = typer.Option(
        False,
        "--simulate-disparity",
        help="Simulate disparate cross-site entities to test retry protocol and disparity reporting.",
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
    1. Detect & encode face (RetinaFace + ArcFace) + Portrait Quality Gate
    2. Simultaneously check multiple sites & evaluate identity consensus (with up to 2 retries)
    3. Finalize on single verified individual or provide full disparity disclosure
    4. Upload multi-site consensus record & Merkle root to Foundry Anvil blockchain
    5. Re-verify data against immutable on-chain record
    """
    console.print(
        Panel.fit(
            "[bold cyan]🛡️  FACE ID + MULTI-SITE IDENTITY CONSENSUS & BLOCKCHAIN PIPELINE[/bold cyan]\n"
            "[dim]Biometric Analysis → Simultaneous Multi-Site Consensus → Immutable On-Chain Registry[/dim]",
            border_style="bright_blue",
        )
    )

    # Allow user to choose file from system if not specified or browse requested
    if choose_file or not image:
        console.print("[cyan]📂 Opening system file dialog to choose an image from your computer...[/cyan]")
        chosen = select_image_file(prompt_if_cancelled=True)
        if not chosen:
            console.print("[bold red]❌ No image file selected. Exiting.[/bold red]")
            raise typer.Exit(code=1)
        image = chosen

    if not os.path.exists(image):
        console.print(f"[bold red]❌ Specified image not found:[/bold red] {image}")
        raise typer.Exit(code=1)

    console.print(f"[bold green]✔ Image selected from system:[/bold green] [bold white]{os.path.abspath(image)}[/bold white]\n")

    resolved_api_key = serpapi_key or os.getenv("SERPAPI_KEY")
    resolved_serper_key = serper_api_key or os.getenv("SERPER_API_KEY")
    has_live_search = bool(resolved_api_key or resolved_serper_key)
    is_demo = demo_search or not has_live_search

    if not is_demo:
        active_providers = []
        if resolved_serper_key:
            active_providers.append("[bold cyan]Serper.dev[/bold cyan] (Google Search API)")
        if resolved_api_key:
            active_providers.append("[bold cyan]SerpAPI[/bold cyan] (Google Lens)")
        console.print(f"[green]✔ Live Search Engine:[/green] {' + '.join(active_providers)}\n")
    elif not demo_search:
        console.print(
            "[yellow]⚠️  Notice: Neither SERPER_API_KEY nor SERPAPI_KEY found in environment. "
            "Proceeding with demo multi-site search mode. Set SERPER_API_KEY in .env for live queries.[/yellow]\n"
        )

    # Resolve target entity hint if provided via CLI or inferred from filename
    resolved_entity = entity_name
    if not resolved_entity:
        import re
        base_fname = os.path.splitext(os.path.basename(image))[0]
        clean_fname = re.sub(r'[_\-\.]+', ' ', base_fname).strip()
        words = [w for w in clean_fname.split() if len(w) > 2 and not w.isdigit()]
        if len(words) >= 2 and not any(w.lower() in ("img", "image", "photo", "pic", "face", "crop", "test") for w in words):
            resolved_entity = " ".join(words).title()

    if resolved_entity:
        console.print(f"[bold cyan]🎯 Explicit Identity Override:[/bold cyan] [bold white]{resolved_entity}[/bold white]\n")
    else:
        console.print("[dim]🔍 Automated Face Identification: System will visually resolve identity from the face image.[/dim]\n")

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
    if face_result.quality:
        face_table.add_row("Blur Variance", f"{face_result.quality.get('blur_variance', 0.0):.2f} (Sharp)")
    face_table.add_row("Cropped Face Path", face_result.cropped_image_path)
    console.print(face_table)

    # -------------------------------------------------------------
    # STEP 2: Simultaneous Multi-Site Search & Identity Consensus
    # -------------------------------------------------------------
    console.print("\n[bold yellow]═══ STEP 2: SIMULTANEOUS MULTI-SITE SEARCH & IDENTITY CONSENSUS ═══[/bold yellow]")
    console.print("[dim]Simultaneously querying multiple platforms to verify all sites finalize on the same person...[/dim]\n")

    search_engine = SearchEngine(
        api_key=resolved_api_key,
        serper_api_key=resolved_serper_key,
    )
    consensus_engine = IdentityConsensusEngine(biometric_threshold=threshold)

    target_platforms = ["Twitter/X", "LinkedIn", "GitHub", "Instagram", "Web"]

    def _execute_search_attempt(retry_lvl: int):
        # In retry attempts, dynamically adapt crop padding to resolve boundary ambiguities
        padding = 0.10 + (retry_lvl * 0.05)
        crop_path = face_engine.analyze(image, crop_padding=padding).cropped_image_path
        return search_engine.search_multi_site(
            image_path=crop_path,
            target_platforms=target_platforms,
            entity_hint=resolved_entity,
            retry_level=retry_lvl,
            demo_mode=is_demo,
            simulate_disparity=simulate_disparity,
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running concurrent multi-site search and cross-verifying identity...", total=None)
        consensus_result = consensus_engine.run_consensus_pipeline(
            search_engine_func=_execute_search_attempt,
            face_engine=face_engine,
            input_face_hash=face_result.face_hash,
            input_embedding=face_result.embedding,
            max_retries=2,  # Try whole pipeline up to 2 times more on disparity
        )
        progress.update(task, completed=True)

    # Render Cross-Site Findings & Consensus Table
    consensus_table = Table(
        title="Cross-Site Identity Consensus Matrix",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )
    consensus_table.add_column("Site / Platform", style="cyan", no_wrap=True)
    consensus_table.add_column("Discovered Profile URL", style="underline blue")
    consensus_table.add_column("Extracted Entity / Handle", style="yellow")
    consensus_table.add_column("Facial Cosine Sim", style="white", justify="center")
    consensus_table.add_column("Consensus Verdict", style="green", justify="center")

    for ev in consensus_result.site_evidences:
        handle_text = ev.detected_name or "Unknown"
        if ev.detected_handle:
            handle_text += f" ({ev.detected_handle})"

        if ev.final_vote == "CONFIRMED_MATCH":
            verdict_badge = "[bold green]✅ SAME PERSON[/bold green]"
            sim_text = f"{ev.biometric_similarity * 100:.1f}%"
        elif ev.final_vote == "NO_PROFILE":
            verdict_badge = "[dim]⚪ NO PROFILE[/dim]"
            sim_text = "N/A"
        else:
            verdict_badge = "[bold red]❌ DIVERGENT[/bold red]"
            sim_text = f"{ev.biometric_similarity * 100:.1f}%"

        # Format URL as terminal hyperlink so clicks always resolve to full URL
        if ev.url and ev.url.startswith("http"):
            url_cell = f"[link={ev.url}]{ev.url}[/link]"
        else:
            url_cell = f"[dim]{ev.url}[/dim]"

        consensus_table.add_row(
            ev.platform,
            url_cell,
            handle_text,
            sim_text,
            verdict_badge,
        )

    console.print(consensus_table)

    # Print unabbreviated direct links for 100% transparent manual verification
    console.print("\n[bold cyan]🔗 Authenticated Direct Profile Links:[/bold cyan]")
    for ev in consensus_result.site_evidences:
        if ev.final_vote == "CONFIRMED_MATCH" and ev.url.startswith("http"):
            console.print(f"  • [bold white]{ev.platform:10s}[/bold white] → [bold underline blue link={ev.url}]{ev.url}[/bold underline blue link={ev.url}] [dim]({ev.detected_name})[/dim]")
        elif ev.final_vote == "DIVERGENT" and ev.url.startswith("http"):
            console.print(f"  • [bold red]{ev.platform:10s}[/bold red] → [underline red link={ev.url}]{ev.url}[/underline red link={ev.url}] [red](Divergent: {ev.detected_name})[/red]")
        elif ev.final_vote == "NO_PROFILE":
            console.print(f"  • [dim]{ev.platform:10s} → No authentic profile found[/dim]")
    console.print()

    # Handle the Disparate Entities Situation or Confirmed Consensus
    if consensus_result.disparate_situation_detected or not consensus_result.consensus_reached:
        reasons = []
        low_bio = [e for e in consensus_result.site_evidences if e.final_vote == "DIVERGENT" and not e.is_biometrically_verified]
        name_diff = [e for e in consensus_result.site_evidences if e.final_vote == "DIVERGENT" and not e.is_name_aligned]
        if low_bio:
            reasons.append(f"{len(low_bio)} platform(s) with biometric similarity below threshold ({threshold*100:.0f}%)")
        if name_diff:
            reasons.append(f"{len(name_diff)} platform(s) with conflicting entity name")
        breakdown_text = "; ".join(reasons) if reasons else "Conflicting identity evidence detected across platforms."

        console.print(
            Panel(
                f"[bold red]⚠️ SITUATION: DISPARATE ENTITIES DETECTED ACROSS CHECKED SITES[/bold red]\n\n"
                f"• [bold]Attempts Executed:[/bold] {consensus_result.attempts_taken} attempt(s)\n"
                f"• [bold]Diagnostic Summary:[/bold] {consensus_result.disparate_summary or 'Multi-site identity consensus could not be verified across quorum.'}\n"
                f"• [bold]Finding Breakdown:[/bold] {breakdown_text}\n"
                f"• [bold]All Discovered Finds:[/bold] Documented in the table above and captured in the forensic report.",
                title="[bold red]Disparate Entities Report[/bold red]",
                border_style="red",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold green]🤝 IDENTITY CONSENSUS REACHED: ALL SITES FINALIZE ON THE SAME PERSON[/bold green]\n\n"
                f"• [bold]Finalized Person:[/bold] [bold white]{consensus_result.finalized_person_name}[/bold white]\n"
                f"• [bold]Verified Sites Agreement:[/bold] {consensus_result.verified_site_count} of {consensus_result.total_sites_checked} platforms in full consensus\n"
                f"• [bold]Consensus Score:[/bold] [bold yellow]{consensus_result.consensus_score * 100:.1f}%[/bold yellow]\n"
                f"• [bold]Cryptographic Merkle Root:[/bold] [bold white]{consensus_result.merkle_root}[/bold white]\n"
                f"• [bold]Verified Profiles:[/bold] {', '.join(consensus_result.verified_platforms)}",
                title="[bold green]Consensus Verification Quorum[/bold green]",
                border_style="green",
            )
        )

    # -------------------------------------------------------------
    # STEP 3: Blockchain Upload (Foundry Anvil)
    # -------------------------------------------------------------
    console.print("\n[bold yellow]═══ STEP 3: UPLOAD CONSENSUS RECORD TO BLOCKCHAIN ═══[/bold yellow]")
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

        progress.update(task, description="Broadcasting recordConsensusMatch transaction to Anvil...")
        metadata_payload = {
            "finalized_person": consensus_result.finalized_person_name,
            "consensus_score": consensus_result.consensus_score,
            "verified_site_count": consensus_result.verified_site_count,
            "total_sites_checked": consensus_result.total_sites_checked,
            "disparate_situation": consensus_result.disparate_situation_detected,
            "detector": face_result.detector,
            "model": face_result.model,
            "confidence": face_result.confidence,
            "attempts_taken": consensus_result.attempts_taken,
        }

        # Submit consensus record to smart contract
        consensus_record_res = blockchain.record_consensus_match(
            face_hash=face_result.face_hash,
            merkle_root=consensus_result.merkle_root,
            entity_name=consensus_result.finalized_person_name,
            platforms=consensus_result.verified_platforms or ["Web"],
            match_urls=consensus_result.verified_urls or [image],
            consensus_score=consensus_result.consensus_score,
            verified_site_count=max(1, consensus_result.verified_site_count),
            metadata=metadata_payload,
        )
        progress.update(task, completed=True)

    chain_table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    chain_table.add_column("Blockchain Field", style="cyan")
    chain_table.add_column("Value", style="green")
    chain_table.add_row("Blockchain Network", f"Foundry Anvil (Chain ID: {blockchain.w3.eth.chain_id})")
    chain_table.add_row("Smart Contract", f"[bold yellow]{consensus_record_res.contract_address}[/bold yellow]")
    chain_table.add_row("Consensus Record ID", f"[bold white]#{consensus_record_res.consensus_id}[/bold white]")
    chain_table.add_row("Transaction Hash", f"[bold green]{consensus_record_res.tx_hash}[/bold green]")
    chain_table.add_row("Block Number", str(consensus_record_res.block_number))
    chain_table.add_row("Gas Used", f"{consensus_record_res.gas_used:,}")
    chain_table.add_row("Merkle Root", f"[bold white]{consensus_record_res.merkle_root}[/bold white]")
    chain_table.add_row("Submitting Wallet", consensus_record_res.recorded_by)
    chain_table.add_row("Block Timestamp", str(consensus_record_res.timestamp))
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
        verification = blockchain.verify_consensus_record(
            consensus_id=consensus_record_res.consensus_id,
            expected_face_hash=face_result.face_hash,
            expected_merkle_root=consensus_result.merkle_root,
        )
        progress.update(task, completed=True)

    ver_table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    ver_table.add_column("Verification Check", style="cyan")
    ver_table.add_column("Result", style="green")
    ver_table.add_row("Face Hash Verified On-Chain", "✅ MATCH" if verification.face_hash_matches else "❌ MISMATCH")
    ver_table.add_row("Merkle Root Verified On-Chain", "✅ MATCH" if verification.merkle_root_matches else "❌ MISMATCH")
    ver_table.add_row("Smart Contract verifyConsensusRecord()", "✅ VERIFIED" if verification.is_verified else "❌ FAILED")
    ver_table.add_row("Finalized Entity Name", verification.entity_name)
    ver_table.add_row("Tamper Evidence Status", "[bold green]AUTHENTIC & TAMPER-EVIDENT[/bold green]" if verification.is_verified else "[bold red]TAMPERED / CORRUPT[/bold red]")
    console.print(ver_table)

    # Final summary panel (distinguish authentic consensus vs disparity audit trail)
    if consensus_result.consensus_reached and not consensus_result.disparate_situation_detected:
        summary_title = "[bold white]Execution Summary[/bold white]"
        summary_border = "green"
        summary_status = "[bold green]CONFIRMED & IMMUTABLE ON BLOCKCHAIN[/bold green]"
        summary_heading = "[bold green]🎉 MULTI-SITE IDENTITY CONSENSUS REACHED & RECORDED ON-CHAIN![/bold green]"
    else:
        summary_title = "[bold yellow]Execution Summary (Disparity Audit)[/bold yellow]"
        summary_border = "yellow"
        summary_status = "[bold yellow]DISPARITY AUDIT RECORDED ON BLOCKCHAIN (QUORUM FAILED)[/bold yellow]"
        summary_heading = "[bold yellow]⚠️ MULTI-SITE PIPELINE FINISHED: DISPARITY AUDIT RECORDED ON-CHAIN[/bold yellow]"

    console.print(
        Panel(
            f"{summary_heading}\n\n"
            f"• [bold]Consensus Record ID:[/bold] #{consensus_record_res.consensus_id}\n"
            f"• [bold]Contract:[/bold] {consensus_record_res.contract_address}\n"
            f"• [bold]Finalized Identity:[/bold] {consensus_result.finalized_person_name}\n"
            f"• [bold]Consensus Score:[/bold] {consensus_result.consensus_score * 100:.1f}%\n"
            f"• [bold]Verified Sites Agreement:[/bold] {consensus_result.verified_site_count} platforms\n"
            f"• [bold]Merkle Root:[/bold] {consensus_record_res.merkle_root}\n"
            f"• [bold]Tx Hash:[/bold] {consensus_record_res.tx_hash}\n"
            f"• [bold]Status:[/bold] {summary_status}",
            title=summary_title,
            border_style=summary_border,
        )
    )

    # Optional JSON export
    if save_json:
        report = {
            "status": "SUCCESS" if consensus_result.consensus_reached else "DISPARITY_AUDITED",
            "face_analysis": face_result.to_dict(),
            "identity_consensus": consensus_result.to_dict(),
            "blockchain_record": consensus_record_res.to_dict(),
            "on_chain_verification": verification.to_dict(),
        }
        with open(save_json, "w") as f:
            json.dump(report, f, indent=2)
        console.print(f"\n[green]📁 Full report saved to:[/green] [bold]{os.path.abspath(save_json)}[/bold]")

    # Intermediate file management: preserve crops by default so user can inspect them
    if cleanup:
        try:
            cleaned = face_engine.cleanup() + cleanup_temporary_files()
            if cleaned > 0:
                console.print(f"[dim]🧹 Cleaned up {cleaned} temporary working file(s).[/dim]")
        except Exception:
            pass
    else:
        console.print(f"[dim]💾 Preserved face crop and intermediate working files in temp_crops/ (use --cleanup to delete).[/dim]")



if __name__ == "__main__":
    app()
