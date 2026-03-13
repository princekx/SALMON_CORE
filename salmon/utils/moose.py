import os
import subprocess
import logging
import uuid
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MooseClient:
    """Client for interacting with the MOOSE archival system.

    Provides high-level methods for data retrieval and query generation.
    """
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the MooseClient.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary.
                Defaults to None.
        """
        self.config = config or {}

    def retrieve(self, query_file: str, moosedir: str, outfile: str, mock: bool = False) -> bool:
        """Execute a MOOSE retrieval command.

        Args:
            query_file (str): Path to the MOOSE query file.
            moosedir (str): MOOSE directory path.
            outfile (str): Local output path for retrieved data.
            mock (bool, optional): If True, mock the retrieval. Defaults to False.

        Returns:
            bool: True if retrieval was successful, False otherwise.
        """
        if mock:
            logger.info(f"MOCK: Successfully retrieved data to {outfile}")
            return True
        else:
            command = f'/opt/moose-client-wrapper/bin/moo select --fill-gaps {query_file} {moosedir} {outfile}'
            logger.info(f"Executing MOOSE command: {command}")
        
            try:
                result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logger.info(f"MOOSE retrieval successful: {result.stdout.decode('utf-8')}")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"MOOSE retrieval failed: {e.stderr.decode('utf-8')}")
                return False
        return True

    def create_query_file(self, template_path: str, output_path: str, replacements: Dict[str, str]):
        """Create a MOOSE query file from a template.

        Args:
            template_path (str): Path to the query template file.
            output_path (str): Path where the generated query should be saved.
            replacements (Dict[str, str]): Key-value pairs for template substitution.
        """
        with open(template_path, 'r') as f_in, open(output_path, 'w') as f_out:
            content = f_in.read()
            for k, v in replacements.items():
                content = content.replace(k, v)
            f_out.write(content)
