#!/usr/bin/env python3
"""
MCP Server for Local Vector RAG Tool
Provides document search capabilities for the LLM VTuber system
"""

import asyncio
import json
from typing import Any, Sequence
from loguru import logger

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp import JSONRPCError
from mcp.types import (
    CallToolResult,
    EmptyResult,
    ListToolsResult,
    TextContent,
    Tool,
)

from vector_search import LocalVectorRAG


class LocalVectorRAGServer:
    """MCP Server for Local Vector RAG functionality"""

    def __init__(self, docs_folder: str = "data/company_docs"):
        """
        Initialize the Local Vector RAG MCP Server

        Args:
            docs_folder: Path to the documents folder
        """
        self.server = Server("local-vector-rag")
        self.rag = LocalVectorRAG(docs_folder)
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            """List available tools"""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="search_documents",
                        description="Search for relevant documents using vector similarity or text matching",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query to find relevant documents"
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "Number of top results to return (default: 3)",
                                    "default": 3
                                }
                            },
                            "required": ["query"]
                        }
                    ),
                    Tool(
                        name="reload_documents",
                        description="Reload all documents from the configured folder",
                        inputSchema={
                            "type": "object",
                            "properties": {}
                        }
                    ),
                    Tool(
                        name="get_document_stats",
                        description="Get statistics about loaded documents",
                        inputSchema={
                            "type": "object",
                            "properties": {}
                        }
                    )
                ]
            )

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> CallToolResult:
            """Handle tool calls"""
            try:
                if name == "search_documents":
                    return await self._handle_search_documents(arguments)
                elif name == "reload_documents":
                    return await self._handle_reload_documents(arguments)
                elif name == "get_document_stats":
                    return await self._handle_get_document_stats(arguments)
                else:
                    raise JSONRPCError(-32601, f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error in tool call {name}: {e}")
                raise JSONRPCError(-32000, f"Tool execution failed: {str(e)}")

    async def _handle_search_documents(self, arguments: dict) -> CallToolResult:
        """Handle document search requests"""
        query = arguments.get("query")
        if not query:
            raise JSONRPCError(-32602, "Missing required parameter: query")
        
        top_k = arguments.get("top_k", 3)
        
        results = self.rag.search(query, top_k)
        
        # Format results for response
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_result = {
                "rank": i,
                "source": result["source"],
                "content": result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"],
                "score": result.get("score", 0),
                "search_type": result.get("search_type", "unknown")
            }
            formatted_results.append(formatted_result)
        
        response_text = f"Found {len(results)} relevant documents for query: '{query}'\n\n"
        
        for result in formatted_results:
            response_text += f"**Document {result['rank']}: {result['source']}**\n"
            response_text += f"Score: {result['score']:.4f} ({result['search_type']} search)\n"
            response_text += f"Content: {result['content']}\n\n"
        
        return CallToolResult(
            content=[TextContent(type="text", text=response_text)]
        )

    async def _handle_reload_documents(self, arguments: dict) -> CallToolResult:
        """Handle document reload requests"""
        self.rag.reload_documents()
        doc_count = self.rag.get_document_count()
        
        return CallToolResult(
            content=[TextContent(
                type="text", 
                text=f"Documents reloaded successfully. Loaded {doc_count} documents."
            )]
        )

    async def _handle_get_document_stats(self, arguments: dict) -> CallToolResult:
        """Handle document statistics requests"""
        doc_count = self.rag.get_document_count()
        supported_extensions = self.rag.get_supported_extensions()
        
        stats_text = f"Document Statistics:\n"
        stats_text += f"- Total documents loaded: {doc_count}\n"
        stats_text += f"- Documents folder: {self.rag.docs_folder}\n"
        stats_text += f"- Supported file types: {', '.join(supported_extensions)}\n"
        stats_text += f"- Vector search available: {'Yes' if self.rag.model else 'No (using text search)'}\n"
        
        return CallToolResult(
            content=[TextContent(type="text", text=stats_text)]
        )

    async def run(self, transport_type: str = "stdio") -> None:
        """Run the MCP server"""
        if transport_type == "stdio":
            from mcp.server.stdio import stdio_server
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="local-vector-rag",
                        server_version="1.0.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    ),
                )


async def main():
    """Main entry point for the MCP server"""
    import sys
    docs_folder = sys.argv[1] if len(sys.argv) > 1 else "data/company_docs"
    
    server = LocalVectorRAGServer(docs_folder)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
