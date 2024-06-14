import { ComponentFixture, TestBed } from '@angular/core/testing';

import { YucaClasificarComponent } from './yuca-clasificar.component';

describe('YucaClasificarComponent', () => {
  let component: YucaClasificarComponent;
  let fixture: ComponentFixture<YucaClasificarComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [YucaClasificarComponent]
    });
    fixture = TestBed.createComponent(YucaClasificarComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
