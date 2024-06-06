import { ComponentFixture, TestBed } from '@angular/core/testing';

import { YucaComponent } from './yuca_ejemplos.component';

describe('YucaComponent', () => {
  let component: YucaComponent;
  let fixture: ComponentFixture<YucaComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [YucaComponent]
    });
    fixture = TestBed.createComponent(YucaComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
