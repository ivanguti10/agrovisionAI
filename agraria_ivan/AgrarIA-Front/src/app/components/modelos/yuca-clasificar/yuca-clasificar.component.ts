import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { DomSanitizer } from '@angular/platform-browser';
import Swal from 'sweetalert2';

@Component({
  selector: 'app-yuca-clasificar',
  templateUrl: './yuca-clasificar.component.html',
  styleUrls: ['./yuca-clasificar.component.scss']
})
export class YucaClasificarComponent {

  mensajeRetorno: string = '';
  archivos: any;
  loading = false;
  graficos: any = [];
  archivosSeleccionados = false;
  mostrarEjemplos = false;

  constructor(private http: HttpClient, public dialog: MatDialog, public sanitizer: DomSanitizer) {}

  getData(event: any) {
    this.archivos = event.target.files;
  }

  cargarPlanta(): void {
    this.loading = true;
    this.mostrarEjemplos = false;

    if (!this.archivos || this.archivos.length === 0) {
      this.archivosSeleccionados = false;
      this.loading = false;
      return;
    }

    this.archivosSeleccionados = true;
    const formData = new FormData();
    for (let i = 0; i < this.archivos.length; i++) {
      formData.append('archivo', this.archivos[i]);
    }

    
    this.http.post<any>('http://127.0.0.1:5000/cargarPlanta', formData).subscribe(
      (response) => {
        // Este bloque se ejecuta si el modelo se carga correctamente
        this.graficos = response;
        this.mensajeRetorno = response.mensaje; // Almacenar el mensaje de duplicados
        this.loading = false;
        this.mostrarEjemplos = true;
        Swal.fire({
          position: 'top-end',
          icon: 'success',
          title: 'Se han cargado las características del modelo',
          showConfirmButton: false,
          timer: 2000,
          backdrop: false
        });
      },
      (error) => {
        // Este bloque se ejecuta en caso de error
        this.loading = false;
        Swal.fire({
          position: 'top-end',
          icon: 'error', // Cambiado a 'error' para reflejar el fallo
          title: 'Error al cargar el modelo',
          text: 'Por favor, inténtalo de nuevo.', // Mensaje adicional opcional
          showConfirmButton: true,
          timer: 5000, // Ajusta el tiempo según necesites
          backdrop: true
        });
      }
    );
  
  }

}
