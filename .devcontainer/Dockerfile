# Use the official Dev Container Python image as the base
FROM mcr.microsoft.com/devcontainers/python:3.10

# Define Neo4j version and edition as arguments for easy updates
ARG NEO4J_VERSION="5.16.0"
ARG NEO4J_EDITION="community" # Use "enterprise" if you have a license
ARG NEO4J_HOME="/opt/neo4j"

# Install necessary packages for Neo4j:
# default-jdk: Neo4j requires Java. This installs OpenJDK.
# wget: To download the Neo4j tarball.
# tar: To extract the tarball.
# sudo: Needed for the postCreateCommand to run neo4j as the 'vscode' user.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        default-jdk \
        wget \
        tar \
        sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and extract Neo4j
# The artifact.php URL is a common way to get direct download links
RUN wget -qO- "https://neo4j.com/artifact.php?name=neo4j-${NEO4J_EDITION}-${NEO4J_VERSION}-unix.tar.gz" | tar -xz --directory /opt \
    && mv /opt/neo4j-${NEO4J_EDITION}-${NEO4J_VERSION} "${NEO4J_HOME}"

# Set ownership of the Neo4j installation directory to the 'vscode' user.
# The base devcontainer images typically create a 'vscode' user (UID/GID 1000)
# This is crucial for Neo4j to run with proper permissions later.
RUN chown -R vscode:vscode "${NEO4J_HOME}"

# Switch to the 'vscode' user for subsequent commands (like configuration)
USER vscode
WORKDIR ${NEO4J_HOME}

# Configure Neo4j:
# - Listen on all network interfaces (0.0.0.0) for HTTP (Browser) and Bolt connections.
# - Ensure authentication is enabled.
# - Set the data directory explicitly (though it's default, good for clarity).
# We use 'sed' to modify the neo4j.conf file.
RUN sed -i 's/#dbms.default_listen_address=0.0.0.0/dbms.default_listen_address=0.0.0.0/' conf/neo4j.conf \
    && sed -i 's/#dbms.connector.bolt.listen_address=0.0.0.0:7687/dbms.connector.bolt.listen_address=0.0.0.0:7687/' conf/neo4j.conf \
    && sed -i 's/#dbms.connector.http.listen_address=0.0.0.0:7474/dbms.connector.http.listen_address=0.0.0.0:7474/' conf/neo4j.conf \
    && sed -i 's/#dbms.security.auth_enabled=true/dbms.security.auth_enabled=true/' conf/neo4j.conf \
    && sed -i 's/#dbms.directories.data=data/dbms.directories.data=data/' conf/neo4j.conf

# Expose ports (for Docker documentation; devcontainer.json handles forwarding)
EXPOSE 7474 7687

# Switch back to root user for postCreateCommand to be able to use sudo if needed,
# though the current postCreateCommand will run as 'vscode' via sudo.
USER root