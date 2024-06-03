Instalar python 3.9
Desinstalar tus librerias e instalar las librerias necesarias

pip3.9 freeze | ForEach-Object {pip uninstall -y $_.split('==')[0]}

pip3.9 install -r paquetes.txt

Insertar funcion especial en _force.py

    #Funcion para que devuelva el html sin crearlo
    def save_html_return(plot):
      html = "<html><head><meta http-equiv='content-type' content='text/html'; charset='utf-8'>\n" + getjs() + "\n</head><body>\n" + plot.html() + "\n</body></html>\n"
      return html
Cambio
