import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-yuca',
  templateUrl: './yuca.component.html',
  styleUrls: ['./yuca.component.scss'],
})
export class YucaComponent {
  constructor(private router: Router) {}

  goToComponentB() {
    this.router.navigate(['/yuca_ejemplos']);
  }
}
