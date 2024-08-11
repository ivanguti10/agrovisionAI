import { Component, OnInit } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { DiseaseInfo } from './model/disease-info.model';  // Importa la interfaz
import { DISEASES_INFO } from './model/diseases';  // Importa el objeto con los datos de las enfermedades
import { EmailService } from '../services/email.service';
import { Plaga } from './model/plaga.model'; // Importa la interfaz Plaga
import { TranslateService } from '@ngx-translate/core';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';
import { environment } from '../../../environments/environment';  // Asegúrate de que la ruta es correcta y que estás importando correctamente el archivo environment.ts

declare var bootstrap: any;

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {
  [x: string]: any;

  imagePreview: string | ArrayBuffer | null = null;
  predictionResult: DiseaseInfo | null = null;
  selectedFile: File | null = null;

  originalImageUrl: SafeUrl | null = null;
  gradCamImageUrl: SafeUrl | null = null;

  name: string = '';
  email: string = '';
  phone: string = '';
  message: string = '';

  plagas: Plaga[] = [];
  selectedFilter: string = 'all';
  filteredPlagas: Plaga[] = [];
  rows: Plaga[][] = [];
  categories: string[] = ['Manzano', 'Arandano', 'Cereza', 'Uva', 'Cítricos', 'Melocotón', 'Pimiento', 'Patata', 'Frambuesa', 'Frijol', 'Calabaza', 'Fresa', 'Tomate'];

  constructor(
    private http: HttpClient,
    private emailService: EmailService,
    private translate: TranslateService,
    private sanitizer: DomSanitizer
  ) {
    translate.setDefaultLang('es');

    const browserLang = translate.getBrowserLang();
    translate.use(browserLang?.match(/en|es/) ? browserLang : 'es');
  }

  switchLanguage(language: string) {
    this.translate.use(language);
  }

  ngOnInit(): void {
    this.getPlagas();
  
    const navCollapse = document.querySelector('.navbar-collapse') as HTMLElement;
    document.querySelector('.navbar-toggler')?.addEventListener('click', () => {
    if (navCollapse.classList.contains('show')) { 
      const bsCollapse = new bootstrap.Collapse(navCollapse, {
        toggle: true
      });
      bsCollapse.hide();
    }
  });

  // Añade este código para cerrar el menú cuando se hace clic en un elemento de la barra de navegación
  const navLinks = document.querySelectorAll('.navbar-nav a');
  navLinks.forEach(link => {
    link.addEventListener('click', () => {
      if (navCollapse.classList.contains('show')) {
        const bsCollapse = new bootstrap.Collapse(navCollapse, {
          toggle: true
        });
        bsCollapse.hide();
      }
    });
  });

  }
  
  closeMenu() {
    const navbarToggler = document.querySelector('.navbar-toggler') as HTMLElement;
    const navbarCollapse = document.querySelector('#navbarResponsive') as HTMLElement;
  
    // Verifica si el menú está desplegado (la clase "show" está presente)
    if (navbarCollapse.classList.contains('show')) {
      navbarToggler.click(); // Simula un clic en el botón de menú para cerrarlo
    }
  }
  

  scrollToSection(event: Event, sectionId: string) {
    event.preventDefault();
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  onFileSelected(event: Event): void {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) {
      this.selectedFile = file;
      const reader = new FileReader();
      reader.onload = () => {
        this.imagePreview = reader.result;
      };
      reader.readAsDataURL(file);
    }
  }

  onUpload() {
    if (this.selectedFile) {
      const formData = new FormData();
      formData.append('file', this.selectedFile, this.selectedFile.name);

      this.http.post(`${environment.apiUrl}/predict`, formData).subscribe(
        (response: any) => {
          console.log('Response:', response);
          this.gradCamImageUrl = this.sanitizer.bypassSecurityTrustUrl('data:image/png;base64,' + response.grad_cam_image);
        },
        (error: HttpErrorResponse) => {
          console.error('Error:', error);
        }
      );
    } else {
      console.log('No file selected');
    }
  }

  getPlagas() {
    this.http.get<Plaga[]>(`${environment.apiUrl}/plagas`).subscribe(
      (data) => {
        this.plagas = data;
        this.filteredPlagas = data;
        this.groupPlagas();
      },
      (error: HttpErrorResponse) => {
        console.error('Error:', error);
      }
    );
  }

  applyFilter() {
    const filter = this.selectedFilter === 'all' ? '' : encodeURIComponent(this.selectedFilter);
    this.http.get<Plaga[]>(`${environment.apiUrl}/plagas?filter=${filter}`).subscribe(
      (data) => {
        this.filteredPlagas = data;
        this.groupPlagas();  // Agrupar las plagas después de filtrar
      },
      (error: HttpErrorResponse) => {
        console.error('Error:', error);
      }
    );
  }

  groupPlagas(): void {
    this.rows = [];
    for (let i = 0; i < this.filteredPlagas.length; i += 3) {
      this.rows.push(this.filteredPlagas.slice(i, i + 3));
    }
  }

  predict(): void {
    if (this.selectedFile) {
      const formData = new FormData();
      formData.append('file', this.selectedFile, this.selectedFile.name);

      this.http.post<any>(`${environment.apiUrl}/predict`, formData).subscribe(
        (response) => {
          const predictedClass = response.class;
          const confidence = response.confidence;
          this.predictionResult = DISEASES_INFO[predictedClass] || null;

          if (this.predictionResult) {
            this.predictionResult.confidence = confidence;
            this.showPredictionModal();
          } else {
            alert(`No se encontró información para la clase predicha: ${predictedClass}`);
          }
        },
        (error: HttpErrorResponse) => {
          console.error('Error:', error);
          alert('Error realizando la predicción: ' + error.message + '. ' + (error.error?.error || ''));
        }
      );
    } else {
      alert('No hay imagen seleccionada.');
    }
  }

  private showPredictionModal(): void {
    const modalElement = document.getElementById('predictionModal');
    if (modalElement) {
      const modal = new bootstrap.Modal(modalElement);
      modal.show();
    }
  }

  refreshPage(): void {
    window.location.reload();
  }

  onSubmit() {
    const formData = {
      name: this.name,
      email: this.email,
      phone: this.phone,
      message: this.message,
    };

    this.emailService.sendEmail(formData).subscribe(
      (response) => {
        console.log('Correo enviado exitosamente', response);
        document.getElementById('submitSuccessMessage')?.classList.remove('d-none');
        document.getElementById('submitErrorMessage')?.classList.add('d-none');
      },
      (error) => {
        console.error('Error al enviar el correo', error);
        document.getElementById('submitErrorMessage')?.classList.remove('d-none');
        document.getElementById('submitSuccessMessage')?.classList.add('d-none');
      }
    );
  }
}
