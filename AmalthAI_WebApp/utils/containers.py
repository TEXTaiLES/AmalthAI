import docker

def container_autostart(client, containers):
    # start automatically all the specified docker containers
    for name in containers:
        try:
            container = client.containers.get(name)
            container.start()
            print(f"Container '{name}' started successfully.")
        except docker.errors.NotFound:
            print(f"Container '{name}' not found.")